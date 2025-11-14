"""
Interface Gráfica para Análise de Ventrículos Laterais
Sistema completo de visualização, segmentação e análise de imagens médicas
Arquivo único com todas as funcionalidades integradas
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import cv2
from scipy import ndimage
from skimage import filters
from scipy.spatial.distance import pdist
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QScrollArea, QGroupBox, QGridLayout, QMenuBar,
                             QAction, QDialog, QSpinBox, QComboBox, QMessageBox,
                             QSplitter, QTabWidget, QFrame)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont, QWheelEvent
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Imports para o classificador
import joblib
import os


# ===============================================================================
# CLASSES DE SEGMENTAÇÃO E ANÁLISE (DO CÓDIGO ORIGINAL)
# ===============================================================================

class VentricleSegmentation:
    """Classe responsável pela segmentação dos ventrículos laterais"""
    
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.img_data = None
        self.slice_idx = None
        self.seed = None
        self.brain_mask = None

    def load_image(self):
        nii_img = nib.load(self.image_path)
        self.img_data = nii_img.get_fdata()
        
        if self.img_data.ndim == 2:
            return self.img_data
        elif self.img_data.ndim == 3:
            self.slice_idx = self.img_data.shape[0] // 2
            return self.img_data[self.slice_idx, :, :]
        else:
            raise ValueError(f"Dimensionalidade não suportada: {self.img_data.ndim}D")

    def normalize_image(self, img):
        img_norm = img - np.min(img)
        img_norm = img_norm / (np.max(img_norm) + 1e-8) * 255
        return img_norm.astype(np.uint8)

    def create_brain_mask(self, img):
        threshold = filters.threshold_otsu(img)
        brain_mask = img > threshold * 0.5
        brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((5, 5)))
        brain_mask = ndimage.binary_closing(brain_mask, structure=np.ones((11, 11)))
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        
        labeled, num_features = ndimage.label(brain_mask)
        if num_features > 0:
            sizes = ndimage.sum(brain_mask, labeled, range(1, num_features + 1))
            max_label = np.argmax(sizes) + 1
            brain_mask = labeled == max_label
        
        brain_mask = ndimage.binary_erosion(brain_mask, structure=np.ones((15, 15)))
        return brain_mask

    def generate_candidate_seeds(self, img, brain_mask, n_candidates=50):
        h, w = img.shape
        candidates = []
        
        img_masked = img.copy()
        img_masked[~brain_mask] = 255
        
        dark_threshold = np.percentile(img_masked[brain_mask], 15)
        dark_regions = (img_masked < dark_threshold) & brain_mask
        
        dark_regions = ndimage.binary_opening(dark_regions, structure=np.ones((3,3)))
        dark_regions = ndimage.binary_closing(dark_regions, structure=np.ones((5,5)))
        
        distance_map = ndimage.distance_transform_edt(dark_regions)
        
        local_max = filters.rank.maximum(distance_map.astype(np.uint8), np.ones((15, 15)))
        seed_candidates = (distance_map == local_max) & (distance_map > 3)
        
        y_coords, x_coords = np.where(seed_candidates)
        
        for y, x in zip(y_coords, x_coords):
            if not brain_mask[y, x]:
                continue
            
            window_size = 15
            y_min = max(0, y - window_size)
            y_max = min(h, y + window_size)
            x_min = max(0, x - window_size)
            x_max = min(w, x + window_size)
            
            window = img[y_min:y_max, x_min:x_max]
            window_mask = brain_mask[y_min:y_max, x_min:x_max]
            
            if np.sum(window_mask) < 50:
                continue
            
            window_values = window[window_mask]
            
            features = {
                'position': (y, x),
                'intensity': float(img[y, x]),
                'mean_intensity': float(np.mean(window_values)),
                'std_intensity': float(np.std(window_values)),
                'homogeneity': float(1.0 / (1.0 + np.std(window_values))),
                'centrality': 1.0 - np.sqrt(((y - h/2)/h)**2 + ((x - w/2)/w)**2),
                'distance_from_edge': float(distance_map[y, x]),
                'distance_from_brain_edge': float(ndimage.distance_transform_edt(brain_mask)[y, x])
            }
            
            candidates.append(features)
        
        return candidates[:n_candidates]

    def score_seed(self, features):
        score = 0.0
        intensity_score = 1.0 - (features['intensity'] / 255.0)
        score += intensity_score * 3.5
        score += features['homogeneity'] * 2.5
        score += features['centrality'] * 2.0
        score += min(features['distance_from_edge'] / 20.0, 1.0) * 1.5
        score += min(features['distance_from_brain_edge'] / 30.0, 1.0) * 4.0
        return score

    def select_best_seed(self, candidates):
        if not candidates:
            return None
        
        scored_candidates = [(self.score_seed(c), c['position']) for c in candidates]
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        return scored_candidates[0][1]

    def region_growing(self, img, seed, threshold=20):
        h, w = img.shape
        segmented = np.zeros((h, w), dtype=bool)
        
        if seed is None:
            return segmented
        
        seed_value = img[seed[0], seed[1]]
        queue = [seed]
        segmented[seed[0], seed[1]] = True
        
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        while queue:
            y, x = queue.pop(0)
            
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < h and 0 <= nx < w and not segmented[ny, nx]:
                    if self.brain_mask is not None and not self.brain_mask[ny, nx]:
                        continue
                    
                    if abs(float(img[ny, nx]) - float(seed_value)) < threshold:
                        segmented[ny, nx] = True
                        queue.append((ny, nx))
        
        segmented = ndimage.binary_opening(segmented, structure=np.ones((3,3)))
        segmented = ndimage.binary_closing(segmented, structure=np.ones((5,5)))
        segmented = ndimage.binary_fill_holes(segmented)
        
        return segmented

    def process_segmentation(self, threshold=20, verbose=False):
        if verbose:
            print(f"\n[Segmentando] {self.image_path.name}")
        
        img = self.load_image()
        img_norm = self.normalize_image(img)
        self.brain_mask = self.create_brain_mask(img_norm)
        candidates = self.generate_candidate_seeds(img_norm, self.brain_mask)
        self.seed = self.select_best_seed(candidates)
        
        if self.seed is None and self.brain_mask is not None:
            y_coords, x_coords = np.where(self.brain_mask)
            if len(y_coords) > 0:
                self.seed = (int(np.mean(y_coords)), int(np.mean(x_coords)))
        
        segmentation = self.region_growing(img_norm, self.seed, threshold)
        
        if verbose:
            seg_pixels = np.sum(segmentation)
            print(f"  ✓ Pixels segmentados: {seg_pixels}")
        
        return segmentation, img_norm


class VentricleDescriptors:
    """Classe responsável pelo cálculo de descritores morfológicos"""
    
    def __init__(self):
        self.descriptors = {}
    
    def calculate_all_descriptors(self, binary_mask, verbose=False):
        if binary_mask.dtype == bool:
            binary_mask = (binary_mask.astype(np.uint8)) * 255
        elif binary_mask.max() <= 1:
            binary_mask = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask = binary_mask.astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return self._empty_descriptors()
        
        main_contour = max(contours, key=cv2.contourArea)
        
        area = self._calculate_area(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        circularity = self._calculate_circularity(area, perimeter)
        eccentricity = self._calculate_eccentricity(main_contour)
        rectangularity = self._calculate_rectangularity(main_contour, area)
        solidity = self._calculate_solidity(main_contour, area)
        diameter = self._calculate_diameter(main_contour)
        
        self.descriptors = {
            'area': area,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'rectangularity': rectangularity,
            'solidity': solidity,
            'diameter': diameter
        }
        
        return self.descriptors
    
    def _calculate_area(self, contour):
        return cv2.contourArea(contour)
    
    def _calculate_circularity(self, area, perimeter):
        if perimeter == 0:
            return 0
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return circularity
    
    def _calculate_eccentricity(self, contour):
        if len(contour) < 5:
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) == 0:
                return 1
            return max(w, h) / min(w, h)
        
        try:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            
            if minor_axis == 0:
                return 1
            
            eccentricity = major_axis / minor_axis
            return eccentricity
        except:
            return 1
    
    def _calculate_rectangularity(self, contour, area):
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        rect_area = width * height
        
        if rect_area == 0:
            return 0
        
        rectangularity = area / rect_area
        return rectangularity
    
    def _calculate_solidity(self, contour, area):
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0
        
        solidity = area / hull_area
        return solidity
    
    def _calculate_diameter(self, contour):
        points = contour.reshape(-1, 2)
        
        if len(points) < 2:
            return 0
        
        if len(points) > 100:
            indices = np.linspace(0, len(points)-1, 100, dtype=int)
            points = points[indices]
        
        distances = pdist(points, metric='euclidean')
        diameter = np.max(distances)
        
        return diameter
    
    def _empty_descriptors(self):
        return {
            'area': 0,
            'circularity': 0,
            'eccentricity': 0,
            'rectangularity': 0,
            'solidity': 0,
            'diameter': 0
        }


# ===============================================================================
# CLASSIFICADOR E REGRESSOR XGBOOST
# ===============================================================================

class BrainClassifier:
    """Classe para carregar e usar o modelo XGBoost treinado"""
    
    def __init__(self, model_path='modelo_xgboost.pkl', scaler_path='scaler.pkl', 
                 label_encoder_path='label_encoder.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = ['area', 'circularity', 'eccentricity', 
                             'rectangularity', 'solidity', 'diameter']
    
    def load_model(self):
        """Carrega modelo, scaler e label encoder salvos"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado: {self.scaler_path}")
        
        if not os.path.exists(self.label_encoder_path):
            raise FileNotFoundError(f"Label encoder não encontrado: {self.label_encoder_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.label_encoder_path)
        
        return True
    
    def predict(self, descriptors):
        """
        Faz predição para descritores de uma imagem
        
        Args:
            descriptors: Dict com os descritores morfológicos
            
        Returns:
            Tupla (classe_predita, probabilidades_dict)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        # Criar DataFrame com os descritores na ordem correta
        features_df = pd.DataFrame([descriptors])[self.feature_names]
        
        # Normalizar
        features_scaled = self.scaler.transform(features_df)
        
        # Predizer
        prediction_encoded = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Decodificar predição
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Criar dicionário de probabilidades
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probabilities[i]
        
        return prediction, prob_dict


class BrainAgeRegressor:
    """Classe para carregar e usar o modelo de regressão de idade"""
    
    def __init__(self, model_path='modelo_xgboost_age.pkl', scaler_path='scaler_age.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = ['area', 'circularity', 'eccentricity', 
                             'rectangularity', 'solidity', 'diameter']
    
    def load_model(self):
        """Carrega modelo e scaler salvos"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado: {self.scaler_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        return True
    
    def predict(self, descriptors):
        """
        Prediz idade para descritores de uma imagem
        
        Args:
            descriptors: Dict com os descritores morfológicos
            
        Returns:
            Idade predita (float)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        # Criar DataFrame com os descritores na ordem correta
        features_df = pd.DataFrame([descriptors])[self.feature_names]
        
        # Normalizar
        features_scaled = self.scaler.transform(features_df)
        
        # Predizer
        age_prediction = self.model.predict(features_scaled)[0]
        
        return age_prediction


# ===============================================================================
# INTERFACE GRÁFICA - WIDGETS CUSTOMIZADOS
# ===============================================================================

class ZoomableLabel(QLabel):
    """Label com capacidade de zoom por scroll"""
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.setMinimumSize(300, 300)
        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        self.clicked.emit()


class ZoomDialog(QDialog):
    """Diálogo para visualização em zoom com scroll"""
    
    def __init__(self, pixmap, title="Zoom", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(800, 800)
        
        # Configurar para tela cheia semi-transparente
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        
        layout = QVBoxLayout()
        
        # Label para a imagem
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border: 3px solid #666;")
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(False)
        scroll.setStyleSheet("background-color: #1a1a1a;")
        
        # Armazenar pixmap original
        self.original_pixmap = pixmap
        self.current_scale = 1.0
        self.update_image()
        
        # Botão fechar
        close_btn = QPushButton("Fechar (ESC)")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("padding: 10px; font-size: 11pt;")
        
        layout.addWidget(scroll)
        layout.addWidget(close_btn)
        self.setLayout(layout)
        
        # Instalar filtro de eventos para capturar scroll
        self.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            # Zoom com scroll
            delta = event.angleDelta().y()
            if delta > 0:
                self.current_scale *= 1.1
            else:
                self.current_scale *= 0.9
            
            # Limitar zoom
            self.current_scale = max(0.1, min(self.current_scale, 10.0))
            self.update_image()
            return True
        return super().eventFilter(obj, event)
    
    def update_image(self):
        """Atualiza a imagem com o scale atual"""
        new_size = self.original_pixmap.size() * self.current_scale
        scaled_pixmap = self.original_pixmap.scaled(
            new_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


class ZoomablePlotCanvas(FigureCanvasQTAgg):
    """Canvas do Matplotlib com zoom"""
    clicked = pyqtSignal()
    
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class SettingsDialog(QDialog):
    """Diálogo de configurações"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurações")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Configuração de fonte
        font_group = QGroupBox("Tamanho da Fonte")
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Tamanho:"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(10)
        font_layout.addWidget(self.font_spin)
        font_group.setLayout(font_layout)
        
        # Configuração de tema
        theme_group = QGroupBox("Tema")
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Modo:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Escuro", "Claro"])
        theme_layout.addWidget(self.theme_combo)
        theme_group.setLayout(theme_layout)
        
        # Botões
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addWidget(font_group)
        layout.addWidget(theme_group)
        layout.addLayout(button_layout)
        self.setLayout(layout)


# ===============================================================================
# JANELA PRINCIPAL
# ===============================================================================

class VentricleAnalysisGUI(QMainWindow):
    """Janela principal da aplicação"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_image_path = None
        self.normalized_image = None
        self.segmentation_mask = None
        self.brain_mask = None
        self.descriptors = None
        self.font_size = 10
        self.theme = "dark"
        
        # Inicializar classificador
        self.classifier = BrainClassifier()
        self.classifier_loaded = False
        
        # Inicializar regressor de idade
        self.regressor = BrainAgeRegressor()
        self.regressor_loaded = False
        
        self.init_ui()
        self.apply_theme()
        self.try_load_classifier()
        self.try_load_regressor()
        
    def init_ui(self):
        """Inicializa a interface"""
        self.setWindowTitle("Análise de Ventrículos Laterais - OASIS Dataset")
        self.setGeometry(100, 100, 1600, 900)
        
        # Menu bar
        self.create_menu_bar()
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)
        
        # Painel esquerdo - Visualização
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Painel direito - Resultados (TABS INDEPENDENTES)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([500, 1100])
        main_layout.addWidget(splitter)
        
    def create_menu_bar(self):
        """Cria a barra de menu"""
        menubar = self.menuBar()
        
        # Menu Arquivo
        file_menu = menubar.addMenu("Arquivo")
        
        open_action = QAction("Abrir Imagem", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Sair", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Configurações
        settings_menu = menubar.addMenu("Configurações")
        
        settings_action = QAction("Preferências", self)
        settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(settings_action)
        
        # Menu Ajuda
        help_menu = menubar.addMenu("Ajuda")
        
        about_action = QAction("Sobre", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_left_panel(self):
        """Cria painel esquerdo com visualização"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Título
        title = QLabel("Visualização da Imagem")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Botão carregar imagem
        self.load_btn = QPushButton("Carregar Imagem (NIfTI, PNG, JPG)")
        self.load_btn.setStyleSheet("padding: 10px; font-size: 11pt;")
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)
        
        # Área de visualização
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.image_label = ZoomableLabel()
        self.image_label.clicked.connect(self.zoom_main_image)
        self.image_label.setText("Nenhuma imagem carregada\n\nClique em 'Carregar Imagem' para começar")
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, stretch=1)
        
        # Botões de ação
        button_layout = QVBoxLayout()
        
        self.segment_btn = QPushButton("Segmentar Ventrículo")
        self.segment_btn.setEnabled(False)
        self.segment_btn.setStyleSheet("padding: 15px; font-size: 12pt; font-weight: bold;")
        self.segment_btn.clicked.connect(self.run_segmentation)
        
        self.classify_btn = QPushButton("Classificar")
        self.classify_btn.setEnabled(False)
        self.classify_btn.setStyleSheet("padding: 15px; font-size: 12pt; font-weight: bold;")
        self.classify_btn.clicked.connect(self.run_classification)
        
        self.regress_btn = QPushButton("Predizer Idade")
        self.regress_btn.setEnabled(False)
        self.regress_btn.setStyleSheet("padding: 15px; font-size: 12pt; font-weight: bold;")
        self.regress_btn.clicked.connect(self.run_regression)
        
        button_layout.addWidget(self.segment_btn)
        button_layout.addWidget(self.classify_btn)
        button_layout.addWidget(self.regress_btn)
        
        layout.addLayout(button_layout)
        
        return panel
        
    def create_right_panel(self):
        """Cria painel direito com TABS INDEPENDENTES para cada operação"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs para cada operação
        self.tabs = QTabWidget()
        
        # Tab 1: Segmentação
        self.segmentation_tab = self.create_segmentation_tab()
        self.tabs.addTab(self.segmentation_tab, "Segmentação")
        
        # Tab 2: Classificação
        self.classification_tab = self.create_classification_tab()
        self.tabs.addTab(self.classification_tab, "Classificação")
        
        # Tab 3: Regressão de Idade
        self.regression_tab = self.create_regression_tab()
        self.tabs.addTab(self.regression_tab, "Predição de Idade")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def create_segmentation_tab(self):
        """Cria tab completa de segmentação com TUDO em uma única aba"""
        tab = QWidget()
        
        # Scroll area para todo o conteúdo
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # ===== SEÇÃO 1: IMAGENS DE SEGMENTAÇÃO (3 em linha) =====
        images_title = QLabel("Resultados da Segmentação")
        images_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(images_title)
        
        images_layout = QHBoxLayout()
        
        # Container 1: Máscara do Cérebro
        brain_container = QWidget()
        brain_vlayout = QVBoxLayout(brain_container)
        brain_label_title = QLabel("Máscara do Cérebro")
        brain_label_title.setAlignment(Qt.AlignCenter)
        brain_label_title.setStyleSheet("font-weight: bold; padding: 5px;")
        self.brain_mask_label = ZoomableLabel()
        self.brain_mask_label.clicked.connect(self.zoom_brain_mask)
        self.brain_mask_label.setText("Aguardando\nsegmentação")
        brain_vlayout.addWidget(brain_label_title)
        brain_vlayout.addWidget(self.brain_mask_label)
        
        # Container 2: Máscara do Ventrículo
        ventricle_container = QWidget()
        ventricle_vlayout = QVBoxLayout(ventricle_container)
        ventricle_label_title = QLabel("Máscara do Ventrículo")
        ventricle_label_title.setAlignment(Qt.AlignCenter)
        ventricle_label_title.setStyleSheet("font-weight: bold; padding: 5px;")
        self.ventricle_mask_label = ZoomableLabel()
        self.ventricle_mask_label.clicked.connect(self.zoom_ventricle_mask)
        self.ventricle_mask_label.setText("Aguardando\nsegmentação")
        ventricle_vlayout.addWidget(ventricle_label_title)
        ventricle_vlayout.addWidget(self.ventricle_mask_label)
        
        # Container 3: Overlay
        overlay_container = QWidget()
        overlay_vlayout = QVBoxLayout(overlay_container)
        overlay_label_title = QLabel("Overlay (Ventrículo em Vermelho)")
        overlay_label_title.setAlignment(Qt.AlignCenter)
        overlay_label_title.setStyleSheet("font-weight: bold; padding: 5px;")
        self.overlay_label = ZoomableLabel()
        self.overlay_label.clicked.connect(self.zoom_overlay)
        self.overlay_label.setText("Aguardando\nsegmentação")
        overlay_vlayout.addWidget(overlay_label_title)
        overlay_vlayout.addWidget(self.overlay_label)
        
        images_layout.addWidget(brain_container)
        images_layout.addWidget(ventricle_container)
        images_layout.addWidget(overlay_container)
        
        content_layout.addLayout(images_layout)
        
        # Separador
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line1)
        
        # ===== SEÇÃO 2: DESCRITORES =====
        desc_title = QLabel("Descritores Morfológicos")
        desc_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(desc_title)
        
        self.descriptors_label = QLabel("Nenhuma segmentação realizada")
        self.descriptors_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.descriptors_label.setStyleSheet("padding: 15px; border: 1px solid #555; background-color: #2a2a2a;")
        self.descriptors_label.setWordWrap(True)
        content_layout.addWidget(self.descriptors_label)
        
        # Separador
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line2)
        
        # ===== SEÇÃO 3: SCATTERPLOTS (GRID 5x3) =====
        scatter_title = QLabel("Scatterplots Comparativos (15 gráficos)")
        scatter_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(scatter_title)
        
        # Container para os scatterplots em grid
        scatter_container = QWidget()
        self.scatter_grid = QGridLayout(scatter_container)
        self.scatter_grid.setSpacing(10)
        self.scatter_canvases = []  # Armazenar referências
        
        # Criar 15 placeholders (5 colunas x 3 linhas)
        for i in range(15):
            row = i // 5
            col = i % 5
            
            placeholder = QLabel(f"Gráfico {i+1}\nAguardando segmentação")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("border: 1px solid #555; min-height: 200px; background-color: #2a2a2a;")
            self.scatter_grid.addWidget(placeholder, row, col)
        
        content_layout.addWidget(scatter_container)
        
        # Adicionar espaçamento no final
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        
        # Layout principal da tab (IMPORTANTE: criar apenas um layout)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        
        return tab
    
    def create_regression_tab(self):
        """Cria tab de regressão de idade"""
        tab = QWidget()
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Título
        title = QLabel("Predição de Idade com XGBoost")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 15px;")
        title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title)
        
        # Status do modelo
        self.regressor_status_label = QLabel()
        self.regressor_status_label.setAlignment(Qt.AlignCenter)
        self.regressor_status_label.setStyleSheet("padding: 10px; border: 1px solid #555; background-color: #2a2a2a;")
        self.update_regressor_status()
        content_layout.addWidget(self.regressor_status_label)
        
        # Separador
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line1)
        
        # Seção: Resultado da Predição
        result_title = QLabel("Idade Predita")
        result_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(result_title)
        
        self.regression_result_label = QLabel("Nenhuma predição realizada.\n\nClique em 'Predizer Idade' para começar.")
        self.regression_result_label.setAlignment(Qt.AlignCenter)
        self.regression_result_label.setStyleSheet("padding: 30px; border: 2px solid #555; background-color: #2a2a2a; font-size: 12pt;")
        self.regression_result_label.setMinimumHeight(200)
        content_layout.addWidget(self.regression_result_label)
        
        # Separador
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line2)
        
        # Seção: Informações sobre a predição
        info_title = QLabel("Sobre a Predição")
        info_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(info_title)
        
        self.regression_info_label = QLabel()
        self.regression_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.regression_info_label.setStyleSheet("padding: 20px; border: 1px solid #555; background-color: #2a2a2a;")
        self.regression_info_label.setWordWrap(True)
        info_text = "<p><b>Como funciona:</b></p>"
        info_text += "<ul>"
        info_text += "<li>O modelo analisa as características morfológicas do ventrículo lateral</li>"
        info_text += "<li>Prediz a idade do paciente no momento do exame</li>"
        info_text += "<li>Baseado em padrões de envelhecimento cerebral</li>"
        info_text += "</ul>"
        info_text += "<p><b>Importante:</b> Esta é uma estimativa baseada apenas em características morfológicas. "
        info_text += "Não substitui avaliação médica profissional.</p>"
        self.regression_info_label.setText(info_text)
        content_layout.addWidget(self.regression_info_label)
        
        # Separador
        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line3)
        
        # Seção: Descritores Usados
        desc_title = QLabel("Descritores Morfológicos Utilizados")
        desc_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(desc_title)
        
        self.regression_descriptors_label = QLabel("Os descritores serão extraídos automaticamente da segmentação.")
        self.regression_descriptors_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.regression_descriptors_label.setStyleSheet("padding: 15px; border: 1px solid #555; background-color: #2a2a2a;")
        self.regression_descriptors_label.setWordWrap(True)
        content_layout.addWidget(self.regression_descriptors_label)
        
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        
        return tab
    
    def create_classification_tab(self):
        """Cria tab de classificação"""
        tab = QWidget()
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Título
        title = QLabel("Classificação com XGBoost")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 15px;")
        title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title)
        
        # Status do modelo
        self.model_status_label = QLabel()
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setStyleSheet("padding: 10px; border: 1px solid #555; background-color: #2a2a2a;")
        self.update_model_status()
        content_layout.addWidget(self.model_status_label)
        
        # Separador
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line1)
        
        # Seção: Resultado da Classificação
        result_title = QLabel("Resultado da Classificação")
        result_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(result_title)
        
        self.classification_result_label = QLabel("Nenhuma classificação realizada.\n\nClique em 'Classificar' para começar.")
        self.classification_result_label.setAlignment(Qt.AlignCenter)
        self.classification_result_label.setStyleSheet("padding: 30px; border: 2px solid #555; background-color: #2a2a2a; font-size: 12pt;")
        self.classification_result_label.setMinimumHeight(200)
        content_layout.addWidget(self.classification_result_label)
        
        # Separador
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line2)
        
        # Seção: Probabilidades
        prob_title = QLabel("Probabilidades por Classe")
        prob_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(prob_title)
        
        self.probabilities_label = QLabel("Aguardando classificação...")
        self.probabilities_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.probabilities_label.setStyleSheet("padding: 20px; border: 1px solid #555; background-color: #2a2a2a;")
        self.probabilities_label.setWordWrap(True)
        content_layout.addWidget(self.probabilities_label)
        
        # Separador
        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setStyleSheet("background-color: #555;")
        content_layout.addWidget(line3)
        
        # Seção: Descritores Usados
        desc_title = QLabel("Descritores Morfológicos Utilizados")
        desc_title.setStyleSheet("font-size: 13pt; font-weight: bold; padding: 10px;")
        content_layout.addWidget(desc_title)
        
        self.classification_descriptors_label = QLabel("Os descritores serão extraídos automaticamente da segmentação.")
        self.classification_descriptors_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.classification_descriptors_label.setStyleSheet("padding: 15px; border: 1px solid #555; background-color: #2a2a2a;")
        self.classification_descriptors_label.setWordWrap(True)
        content_layout.addWidget(self.classification_descriptors_label)
        
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        
        return tab
    
    def load_image(self):
        """Carrega imagem do disco"""
        file_filter = "Imagens (*.nii *.nii.gz *.png *.jpg *.jpeg);;NIfTI (*.nii *.nii.gz);;PNG (*.png);;JPEG (*.jpg *.jpeg)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", file_filter)
        
        if not file_path:
            return
            
        try:
            file_path = Path(file_path)
            self.current_image_path = file_path
            
            # Carregar imagem conforme formato
            if file_path.suffix in ['.nii', '.gz']:
                nii_img = nib.load(str(file_path))
                img_data = nii_img.get_fdata()
                
                # Se 3D, pegar corte sagital central
                if img_data.ndim == 3:
                    slice_idx = img_data.shape[0] // 2
                    self.current_image = img_data[slice_idx, :, :]
                else:
                    self.current_image = img_data
                    
            else:  # PNG ou JPG
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                self.current_image = img
            
            # Normalizar para visualização
            self.normalized_image = self.normalize_image(self.current_image)
            
            # Mostrar imagem
            self.display_image(self.normalized_image, self.image_label)
            
            # Habilitar botão de segmentação
            self.segment_btn.setEnabled(True)
            
            # Habilitar classificação se modelo estiver carregado
            if self.classifier_loaded:
                self.classify_btn.setEnabled(True)
            
            # Habilitar regressão se modelo estiver carregado
            if self.regressor_loaded:
                self.regress_btn.setEnabled(True)
            
            # Resetar resultados
            self.reset_results()
            
            self.statusBar().showMessage(f"Imagem carregada: {file_path.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar imagem:\n{str(e)}")
            
    def normalize_image(self, img):
        """Normaliza imagem para [0, 255]"""
        img_norm = img - np.min(img)
        img_norm = img_norm / (np.max(img_norm) + 1e-8) * 255
        return img_norm.astype(np.uint8)
        
    def display_image(self, img_array, label, colormap=None):
        """Exibe array numpy em QLabel"""
        if img_array is None:
            return
        
        # Fazer cópia para não modificar o original
        img_array = img_array.copy()
            
        # Converter para uint8 se necessário
        if img_array.dtype == bool:
            img_array = (img_array.astype(np.uint8)) * 255
        elif img_array.max() <= 1:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        # Garantir que é contíguo em memória
        img_array = np.ascontiguousarray(img_array)
        
        h, w = img_array.shape
        
        # Aplicar colormap se especificado
        if colormap is not None:
            img_colored = cv2.applyColorMap(img_array, colormap)
            img_colored = np.ascontiguousarray(img_colored)
            bytes_per_line = 3 * w
            q_img = QImage(img_colored.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            q_img = q_img.rgbSwapped()
        else:
            bytes_per_line = w
            q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
        
        pixmap = QPixmap.fromImage(q_img)
        
        # Redimensionar mantendo aspecto
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Armazenar pixmap original para zoom
        label.original_pixmap = pixmap
        
    def display_overlay(self, img_rgb, label):
        """Exibe imagem RGB em QLabel"""
        if img_rgb is None:
            return
        
        # Garantir formato correto
        img_rgb = np.ascontiguousarray(img_rgb)
        h, w, c = img_rgb.shape
        bytes_per_line = 3 * w
        
        # Criar QImage a partir dos dados
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)
        
        # Redimensionar mantendo aspecto
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Armazenar pixmap original para zoom
        label.original_pixmap = pixmap
    
    def zoom_main_image(self):
        """Abre zoom da imagem principal"""
        if hasattr(self.image_label, 'original_pixmap'):
            dialog = ZoomDialog(self.image_label.original_pixmap, "Imagem Original - Use scroll para zoom", self)
            dialog.exec_()
    
    def zoom_brain_mask(self):
        """Abre zoom da máscara do cérebro"""
        if hasattr(self.brain_mask_label, 'original_pixmap'):
            dialog = ZoomDialog(self.brain_mask_label.original_pixmap, "Máscara do Cérebro - Use scroll para zoom", self)
            dialog.exec_()
    
    def zoom_ventricle_mask(self):
        """Abre zoom da máscara do ventrículo"""
        if hasattr(self.ventricle_mask_label, 'original_pixmap'):
            dialog = ZoomDialog(self.ventricle_mask_label.original_pixmap, "Máscara do Ventrículo - Use scroll para zoom", self)
            dialog.exec_()
    
    def zoom_overlay(self):
        """Abre zoom do overlay"""
        if hasattr(self.overlay_label, 'original_pixmap'):
            dialog = ZoomDialog(self.overlay_label.original_pixmap, "Overlay - Use scroll para zoom", self)
            dialog.exec_()
            
    def run_segmentation(self):
        """Executa segmentação do ventrículo"""
        if self.current_image is None:
            return
            
        try:
            self.statusBar().showMessage("Segmentando ventrículo...")
            QApplication.processEvents()
            
            # Salvar temporariamente se for PNG/JPG
            if self.current_image_path.suffix in ['.png', '.jpg', '.jpeg']:
                temp_nii = Path("temp_image.nii.gz")
                nii_img = nib.Nifti1Image(self.current_image, np.eye(4))
                nib.save(nii_img, str(temp_nii))
                seg_path = temp_nii
            else:
                seg_path = self.current_image_path
            
            # Executar segmentação
            segmenter = VentricleSegmentation(str(seg_path))
            self.segmentation_mask, self.normalized_image = segmenter.process_segmentation(
                threshold=20,
                verbose=False
            )
            self.brain_mask = segmenter.brain_mask
            
            # Calcular descritores
            descriptor_extractor = VentricleDescriptors()
            self.descriptors = descriptor_extractor.calculate_all_descriptors(
                self.segmentation_mask,
                verbose=False
            )
            
            # Exibir resultados
            self.display_results()
            
            # Gerar scatterplots
            self.generate_scatterplots()
            
            self.statusBar().showMessage("Segmentação concluída!")
            
            # Limpar temporário
            if self.current_image_path.suffix in ['.png', '.jpg', '.jpeg']:
                temp_nii.unlink(missing_ok=True)
                
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na segmentação:\n{str(e)}")
            self.statusBar().showMessage("Erro na segmentação")
            
    def display_results(self):
        """Exibe resultados da segmentação"""
        try:
            print("DEBUG: Iniciando display_results")
            print(f"DEBUG: Brain mask shape: {self.brain_mask.shape if self.brain_mask is not None else 'None'}")
            print(f"DEBUG: Segmentation mask shape: {self.segmentation_mask.shape if self.segmentation_mask is not None else 'None'}")
            
            # Máscara do cérebro
            brain_display = (self.brain_mask.astype(np.uint8) * 255).copy()
            self.display_image(brain_display, self.brain_mask_label)
            print("DEBUG: Brain mask exibida")
            
            # Máscara do ventrículo
            ventricle_display = (self.segmentation_mask.astype(np.uint8) * 255).copy()
            self.display_image(ventricle_display, self.ventricle_mask_label)
            print("DEBUG: Ventricle mask exibida")
            
            # Overlay
            overlay = self.create_overlay()
            self.display_overlay(overlay, self.overlay_label)
            print("DEBUG: Overlay exibido")
            
            # Forçar atualização visual
            self.brain_mask_label.update()
            self.ventricle_mask_label.update()
            self.overlay_label.update()
            QApplication.processEvents()
            
            # Descritores
            desc_text = "<h3 style='margin: 0; padding: 10px 0;'>Descritores Morfológicos Calculados:</h3>"
            desc_text += "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
            desc_text += "<tr style='background-color: #444;'><th style='padding: 10px; border: 1px solid #666; text-align: left;'>Descritor</th><th style='padding: 10px; border: 1px solid #666; text-align: right;'>Valor</th></tr>"
            
            desc_names = {
                'area': 'Área (pixels)',
                'circularity': 'Circularidade',
                'eccentricity': 'Excentricidade',
                'rectangularity': 'Retangularidade',
                'solidity': 'Solidez',
                'diameter': 'Diâmetro (pixels)'
            }
            
            for key, name in desc_names.items():
                value = self.descriptors.get(key, 0)
                desc_text += f"<tr><td style='padding: 10px; border: 1px solid #666;'><b>{name}</b></td><td style='padding: 10px; border: 1px solid #666; text-align: right; font-family: monospace;'>{value:.4f}</td></tr>"
            
            desc_text += "</table>"
            
            self.descriptors_label.setText(desc_text)
            self.descriptors_label.update()
            print("DEBUG: Descritores exibidos")
            
            # Mudar para a aba de segmentação
            self.tabs.setCurrentIndex(0)
            
        except Exception as e:
            print(f"DEBUG: Erro em display_results: {e}")
            import traceback
            traceback.print_exc()
    
    def try_load_classifier(self):
        """Tenta carregar o modelo de classificação"""
        try:
            self.classifier.load_model()
            self.classifier_loaded = True
            self.statusBar().showMessage("Modelo de classificação carregado com sucesso!", 3000)
        except FileNotFoundError as e:
            self.classifier_loaded = False
            print(f"Modelo não encontrado: {e}")
            self.statusBar().showMessage("Modelo de classificação não encontrado", 3000)
        except Exception as e:
            self.classifier_loaded = False
            print(f"Erro ao carregar modelo: {e}")
    
    def try_load_regressor(self):
        """Tenta carregar o modelo de regressão de idade"""
        try:
            self.regressor.load_model()
            self.regressor_loaded = True
            self.statusBar().showMessage("Modelo de regressão carregado com sucesso!", 3000)
        except FileNotFoundError as e:
            self.regressor_loaded = False
            print(f"Modelo de regressão não encontrado: {e}")
            self.statusBar().showMessage("Modelo de regressão não encontrado", 3000)
        except Exception as e:
            self.regressor_loaded = False
            print(f"Erro ao carregar modelo de regressão: {e}")
    
    def update_model_status(self):
        """Atualiza o status do modelo na interface"""
        if self.classifier_loaded:
            status_text = "<h3 style='color: #4CAF50;'>✓ Modelo Carregado</h3>"
            status_text += "<p>O modelo XGBoost está pronto para classificar imagens.</p>"
            status_text += f"<p><b>Classes disponíveis:</b> {', '.join(self.classifier.label_encoder.classes_)}</p>"
        else:
            status_text = "<h3 style='color: #f44336;'>✗ Modelo Não Encontrado</h3>"
            status_text += "<p>Os arquivos do modelo não foram encontrados:</p>"
            status_text += "<ul>"
            status_text += "<li>modelo_xgboost.pkl</li>"
            status_text += "<li>scaler.pkl</li>"
            status_text += "<li>label_encoder.pkl</li>"
            status_text += "</ul>"
            status_text += "<p>Execute o treinamento primeiro ou coloque os arquivos no diretório.</p>"
        
        self.model_status_label.setText(status_text)
    
    def update_regressor_status(self):
        """Atualiza o status do modelo de regressão na interface"""
        if self.regressor_loaded:
            status_text = "<h3 style='color: #4CAF50;'>✓ Modelo Carregado</h3>"
            status_text += "<p>O modelo XGBoost Regressor está pronto para predizer idades.</p>"
            status_text += "<p><b>Características usadas:</b> Descritores morfológicos do ventrículo</p>"
        else:
            status_text = "<h3 style='color: #f44336;'>✗ Modelo Não Encontrado</h3>"
            status_text += "<p>Os arquivos do modelo não foram encontrados:</p>"
            status_text += "<ul>"
            status_text += "<li>modelo_xgboost_age.pkl</li>"
            status_text += "<li>scaler_age.pkl</li>"
            status_text += "</ul>"
            status_text += "<p>Execute o treinamento primeiro ou coloque os arquivos no diretório.</p>"
        
        self.regressor_status_label.setText(status_text)
    
    def run_classification(self):
        """Executa classificação da imagem atual"""
        if not self.classifier_loaded:
            QMessageBox.warning(self, "Modelo Não Carregado", 
                              "O modelo de classificação não está disponível.\n\n"
                              "Certifique-se de que os arquivos estão no diretório:\n"
                              "- modelo_xgboost.pkl\n"
                              "- scaler.pkl\n"
                              "- label_encoder.pkl")
            return
        
        if self.current_image is None:
            QMessageBox.warning(self, "Sem Imagem", 
                              "Carregue uma imagem primeiro!")
            return
        
        try:
            self.statusBar().showMessage("Classificando imagem...")
            QApplication.processEvents()
            
            # Se ainda não temos descritores, fazer segmentação primeiro
            if self.descriptors is None or self.descriptors.get('area', 0) == 0:
                QMessageBox.information(self, "Segmentação Necessária",
                                      "A imagem será segmentada primeiro para extrair os descritores.")
                self.run_segmentation()
                
                # Verificar se a segmentação foi bem-sucedida
                if self.descriptors is None or self.descriptors.get('area', 0) == 0:
                    QMessageBox.warning(self, "Erro na Segmentação",
                                      "Não foi possível segmentar o ventrículo.\n"
                                      "A classificação não pode ser realizada.")
                    return
            
            # Fazer predição
            prediction, probabilities = self.classifier.predict(self.descriptors)
            
            # Exibir resultados
            self.display_classification_results(prediction, probabilities)
            
            # Mudar para aba de classificação
            self.tabs.setCurrentIndex(1)
            
            self.statusBar().showMessage("Classificação concluída!")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na classificação:\n{str(e)}")
            self.statusBar().showMessage("Erro na classificação")
            import traceback
            traceback.print_exc()
    
    def display_classification_results(self, prediction, probabilities):
        """Exibe resultados da classificação"""
        # Resultado principal
        result_text = f"<h2 style='text-align: center; padding: 20px;'>Classificação: "
        
        # Colorir resultado baseado na classe
        if prediction.lower() == 'demented':
            result_text += f"<span style='color: #f44336; font-weight: bold;'>{prediction.upper()}</span>"
        elif prediction.lower() == 'nondemented':
            result_text += f"<span style='color: #4CAF50; font-weight: bold;'>{prediction.upper()}</span>"
        else:
            result_text += f"<span style='color: #FFC107; font-weight: bold;'>{prediction.upper()}</span>"
        
        result_text += "</h2>"
        
        # Confiança
        confidence = probabilities[prediction] * 100
        result_text += f"<p style='text-align: center; font-size: 14pt;'>Confiança: <b>{confidence:.2f}%</b></p>"
        
        self.classification_result_label.setText(result_text)
        
        # Probabilidades
        prob_text = "<h3 style='margin: 0; padding: 10px 0;'>Probabilidades:</h3>"
        prob_text += "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
        prob_text += "<tr style='background-color: #444;'><th style='padding: 10px; border: 1px solid #666; text-align: left;'>Classe</th><th style='padding: 10px; border: 1px solid #666; text-align: right;'>Probabilidade</th><th style='padding: 10px; border: 1px solid #666;'>Barra</th></tr>"
        
        # Ordenar por probabilidade
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, prob in sorted_probs:
            prob_percent = prob * 100
            bar_width = int(prob * 200)  # Barra de até 200px
            
            # Cor da barra
            if class_name.lower() == 'demented':
                bar_color = '#f44336'
            elif class_name.lower() == 'nondemented':
                bar_color = '#4CAF50'
            else:
                bar_color = '#FFC107'
            
            prob_text += f"<tr>"
            prob_text += f"<td style='padding: 10px; border: 1px solid #666;'><b>{class_name}</b></td>"
            prob_text += f"<td style='padding: 10px; border: 1px solid #666; text-align: right; font-family: monospace;'>{prob_percent:.2f}%</td>"
            prob_text += f"<td style='padding: 10px; border: 1px solid #666;'><div style='background-color: {bar_color}; width: {bar_width}px; height: 20px; border-radius: 3px;'></div></td>"
            prob_text += f"</tr>"
        
        prob_text += "</table>"
        
        self.probabilities_label.setText(prob_text)
        
        # Descritores usados
        desc_text = "<h3 style='margin: 0; padding: 10px 0;'>Descritores Utilizados na Classificação:</h3>"
        desc_text += "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
        desc_text += "<tr style='background-color: #444;'><th style='padding: 10px; border: 1px solid #666; text-align: left;'>Descritor</th><th style='padding: 10px; border: 1px solid #666; text-align: right;'>Valor</th></tr>"
        
        desc_names = {
            'area': 'Área (pixels)',
            'circularity': 'Circularidade',
            'eccentricity': 'Excentricidade',
            'rectangularity': 'Retangularidade',
            'solidity': 'Solidez',
            'diameter': 'Diâmetro (pixels)'
        }
        
        for key, name in desc_names.items():
            value = self.descriptors.get(key, 0)
            desc_text += f"<tr><td style='padding: 10px; border: 1px solid #666;'><b>{name}</b></td><td style='padding: 10px; border: 1px solid #666; text-align: right; font-family: monospace;'>{value:.4f}</td></tr>"
        
        desc_text += "</table>"
        
        self.classification_descriptors_label.setText(desc_text)
    
    def run_regression(self):
        """Executa predição de idade da imagem atual"""
        if not self.regressor_loaded:
            QMessageBox.warning(self, "Modelo Não Carregado", 
                              "O modelo de regressão não está disponível.\n\n"
                              "Certifique-se de que os arquivos estão no diretório:\n"
                              "- modelo_xgboost_age.pkl\n"
                              "- scaler_age.pkl")
            return
        
        if self.current_image is None:
            QMessageBox.warning(self, "Sem Imagem", 
                              "Carregue uma imagem primeiro!")
            return
        
        try:
            self.statusBar().showMessage("Predizendo idade...")
            QApplication.processEvents()
            
            # Se ainda não temos descritores, fazer segmentação primeiro
            if self.descriptors is None or self.descriptors.get('area', 0) == 0:
                QMessageBox.information(self, "Segmentação Necessária",
                                      "A imagem será segmentada primeiro para extrair os descritores.")
                self.run_segmentation()
                
                # Verificar se a segmentação foi bem-sucedida
                if self.descriptors is None or self.descriptors.get('area', 0) == 0:
                    QMessageBox.warning(self, "Erro na Segmentação",
                                      "Não foi possível segmentar o ventrículo.\n"
                                      "A predição de idade não pode ser realizada.")
                    return
            
            # Fazer predição
            predicted_age = self.regressor.predict(self.descriptors)
            
            # Exibir resultados
            self.display_regression_results(predicted_age)
            
            # Mudar para aba de regressão
            self.tabs.setCurrentIndex(2)
            
            self.statusBar().showMessage("Predição de idade concluída!")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na predição de idade:\n{str(e)}")
            self.statusBar().showMessage("Erro na predição")
            import traceback
            traceback.print_exc()
    
    def display_regression_results(self, predicted_age):
        """Exibe resultados da predição de idade"""
        # Resultado principal
        result_text = "<div style='text-align: center; padding: 30px;'>"
        result_text += "<h2 style='margin-bottom: 10px;'>Idade Predita</h2>"
        result_text += f"<h1 style='color: #2196F3; font-size: 48pt; margin: 20px 0;'>{predicted_age:.1f}</h1>"
        result_text += "<h3>anos</h3>"
        result_text += "</div>"
        
        # Adicionar faixa etária
        if predicted_age < 30:
            age_range = "Adulto Jovem"
            icon = "👶"
        elif predicted_age < 50:
            age_range = "Adulto"
            icon = "👨"
        elif predicted_age < 70:
            age_range = "Adulto Maduro"
            icon = "👴"
        else:
            age_range = "Idoso"
            icon = "👵"
        
        result_text += f"<p style='text-align: center; font-size: 14pt;'>{icon} <b>{age_range}</b></p>"
        
        self.regression_result_label.setText(result_text)
        
        # Descritores usados
        desc_text = "<h3 style='margin: 0; padding: 10px 0;'>Descritores Utilizados na Predição:</h3>"
        desc_text += "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
        desc_text += "<tr style='background-color: #444;'><th style='padding: 10px; border: 1px solid #666; text-align: left;'>Descritor</th><th style='padding: 10px; border: 1px solid #666; text-align: right;'>Valor</th></tr>"
        
        desc_names = {
            'area': 'Área (pixels)',
            'circularity': 'Circularidade',
            'eccentricity': 'Excentricidade',
            'rectangularity': 'Retangularidade',
            'solidity': 'Solidez',
            'diameter': 'Diâmetro (pixels)'
        }
        
        for key, name in desc_names.items():
            value = self.descriptors.get(key, 0)
            desc_text += f"<tr><td style='padding: 10px; border: 1px solid #666;'><b>{name}</b></td><td style='padding: 10px; border: 1px solid #666; text-align: right; font-family: monospace;'>{value:.4f}</td></tr>"
        
        desc_text += "</table>"
        
        self.regression_descriptors_label.setText(desc_text)
        
    def create_overlay(self):
        """Cria overlay da segmentação sobre a imagem original"""
        # Fazer cópia contígua da imagem normalizada
        img_base = np.ascontiguousarray(self.normalized_image.copy())
        overlay = cv2.cvtColor(img_base, cv2.COLOR_GRAY2BGR)
        
        # Colorir ventrículo em vermelho
        mask_colored = np.zeros_like(overlay)
        mask_colored[self.segmentation_mask] = [0, 0, 255]  # Vermelho em BGR
        
        # Blend
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Converter BGR para RGB para exibição correta
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = np.ascontiguousarray(overlay)
        
        return overlay
        
    def generate_scatterplots(self):
        """Gera TODOS os 15 scatterplots em grid 5x3"""
        # Limpar scatterplots anteriores
        for i in reversed(range(self.scatter_grid.count())): 
            widget = self.scatter_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        self.scatter_canvases = []
        
        # Verificar se existe CSV com dados
        csv_path = Path("ventricle_descriptors_full.csv")
        if not csv_path.exists():
            for i in range(15):
                row = i // 5
                col = i % 5
                label = QLabel(f"Gráfico {i+1}\nCSV não encontrado")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid #555; min-height: 180px; background-color: #2a2a2a;")
                self.scatter_grid.addWidget(label, row, col)
            return
        
        try:
            df = pd.read_csv(csv_path)
            df_valid = df[(df['area'] > 0) & (df['Group'].notna())]
            
            # Cores
            color_map = {'Converted': 'black', 'Nondemented': 'blue', 'Demented': 'red'}
            
            features = ['area', 'circularity', 'eccentricity', 'rectangularity', 'solidity', 'diameter']
            pairs = list(combinations(features, 2))  # Todas as 15 combinações
            
            # Gerar TODOS os 15 scatterplots
            for idx, (feat_x, feat_y) in enumerate(pairs):
                row = idx // 5
                col = idx % 5
                
                # Criar canvas clicável
                canvas = ZoomablePlotCanvas(self, width=3.5, height=2.5, dpi=80)
                canvas.clicked.connect(lambda c=canvas: self.zoom_scatterplot(c))
                
                # Plotar dados do dataset
                for group, color in color_map.items():
                    df_group = df_valid[df_valid['Group'] == group]
                    if len(df_group) > 0:
                        canvas.axes.scatter(
                            df_group[feat_x], df_group[feat_y],
                            c=color, label=group, alpha=0.5, s=30,
                            edgecolors='black', linewidth=0.3
                        )
                
                # Marcar imagem atual
                if self.descriptors:
                    canvas.axes.scatter(
                        self.descriptors[feat_x], self.descriptors[feat_y],
                        c='yellow', marker='*', s=200,
                        edgecolors='black', linewidth=1.5,
                        label='Atual', zorder=10
                    )
                
                canvas.axes.set_xlabel(feat_x.capitalize(), fontsize=8, fontweight='bold')
                canvas.axes.set_ylabel(feat_y.capitalize(), fontsize=8, fontweight='bold')
                canvas.axes.set_title(f'{feat_x.capitalize()} vs {feat_y.capitalize()}', 
                                     fontsize=9, fontweight='bold')
                canvas.axes.legend(fontsize=6, loc='best')
                canvas.axes.grid(True, alpha=0.3)
                canvas.axes.tick_params(labelsize=7)
                canvas.figure.tight_layout()
                
                # Adicionar ao grid
                self.scatter_grid.addWidget(canvas, row, col)
                self.scatter_canvases.append(canvas)
                
        except Exception as e:
            for i in range(15):
                row = i // 5
                col = i % 5
                label = QLabel(f"Gráfico {i+1}\nErro: {str(e)[:30]}")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid #555; min-height: 180px; background-color: #2a2a2a;")
                self.scatter_grid.addWidget(label, row, col)
    
    def zoom_scatterplot(self, canvas):
        """Abre zoom de um scatterplot"""
        # Criar figura maior
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Copiar dados do canvas original
        for line in canvas.axes.get_lines():
            ax.plot(line.get_xdata(), line.get_ydata(), 
                   marker=line.get_marker(),
                   color=line.get_color(),
                   linestyle='',
                   markersize=line.get_markersize(),
                   label=line.get_label())
        
        for collection in canvas.axes.collections:
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                ax.scatter(offsets[:, 0], offsets[:, 1],
                          c=collection.get_facecolors(),
                          s=collection.get_sizes() * 2,
                          alpha=collection.get_alpha(),
                          edgecolors=collection.get_edgecolors(),
                          linewidths=collection.get_linewidths())
        
        ax.set_xlabel(canvas.axes.get_xlabel(), fontsize=12, fontweight='bold')
        ax.set_ylabel(canvas.axes.get_ylabel(), fontsize=12, fontweight='bold')
        ax.set_title(canvas.axes.get_title(), fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Salvar em buffer e criar pixmap
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        import PIL.Image
        pil_img = PIL.Image.open(buf)
        pil_img = pil_img.convert('RGB')
        img_array = np.array(pil_img)
        
        h, w, c = img_array.shape
        bytes_per_line = 3 * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)
        
        dialog = ZoomDialog(pixmap, "Scatterplot - Use scroll para zoom", self)
        dialog.exec_()
        
        buf.close()
            
    def reset_results(self):
        """Reseta visualizações de resultados"""
        # Segmentação
        self.brain_mask_label.setText("Aguardando\nsegmentação")
        self.ventricle_mask_label.setText("Aguardando\nsegmentação")
        self.overlay_label.setText("Aguardando\nsegmentação")
        self.descriptors_label.setText("Nenhuma segmentação realizada")
        
        # Classificação
        self.classification_result_label.setText("Nenhuma classificação realizada.\n\nClique em 'Classificar' para começar.")
        self.probabilities_label.setText("Aguardando classificação...")
        self.classification_descriptors_label.setText("Os descritores serão extraídos automaticamente da segmentação.")
        
        # Regressão
        self.regression_result_label.setText("Nenhuma predição realizada.\n\nClique em 'Predizer Idade' para começar.")
        self.regression_descriptors_label.setText("Os descritores serão extraídos automaticamente da segmentação.")
        
        # Limpar scatterplots
        for i in reversed(range(self.scatter_grid.count())): 
            widget = self.scatter_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Recriar placeholders
        for i in range(15):
            row = i // 5
            col = i % 5
            placeholder = QLabel(f"Gráfico {i+1}\nAguardando segmentação")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("border: 1px solid #555; min-height: 200px; background-color: #2a2a2a;")
            self.scatter_grid.addWidget(placeholder, row, col)
            
    def open_settings(self):
        """Abre diálogo de configurações"""
        dialog = SettingsDialog(self)
        dialog.font_spin.setValue(self.font_size)
        dialog.theme_combo.setCurrentText("Escuro" if self.theme == "dark" else "Claro")
        
        if dialog.exec_() == QDialog.Accepted:
            self.font_size = dialog.font_spin.value()
            new_theme = "dark" if dialog.theme_combo.currentText() == "Escuro" else "light"
            
            if new_theme != self.theme:
                self.theme = new_theme
                self.apply_theme()
                
            self.apply_font_size()
            
    def apply_theme(self):
        """Aplica tema escuro ou claro"""
        if self.theme == "dark":
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #3d3d3d;
                    border: 1px solid #555;
                    padding: 8px;
                    border-radius: 4px;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:disabled {
                    background-color: #2a2a2a;
                    color: #666;
                }
                QLabel {
                    color: #ffffff;
                }
                QGroupBox {
                    border: 1px solid #555;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    color: #ffffff;
                }
                QTabWidget::pane {
                    border: 1px solid #555;
                }
                QTabBar::tab {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    padding: 10px 20px;
                    border: 1px solid #555;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #4a4a4a;
                    border-bottom: 2px solid #0078d7;
                }
                QScrollArea {
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #f0f0f0;
                    color: #000000;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    border: 1px solid #999;
                    padding: 8px;
                    border-radius: 4px;
                    color: #000000;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QPushButton:disabled {
                    background-color: #f5f5f5;
                    color: #999;
                }
                QLabel {
                    color: #000000;
                }
            """)
            
    def apply_font_size(self):
        """Aplica tamanho de fonte"""
        font = QFont()
        font.setPointSize(self.font_size)
        QApplication.setFont(font)
        
    def show_about(self):
        """Mostra diálogo sobre"""
        QMessageBox.about(self, "Sobre", 
                         "Análise de Ventrículos Laterais\n\n"
                         "Sistema de visualização e segmentação\n"
                         "de ventrículos em imagens de RM.\n\n"
                         "Dataset: OASIS Longitudinal\n"
                         "Desenvolvido para análise de Alzheimer\n\n"
                         "Clique nas imagens para zoom com scroll")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = VentricleAnalysisGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()