# ===============================================================================
# ANÁLISE COMPLETA DE VENTRÍCULOS LATERAIS - DATASET OASIS
# Segmentação + Extração de Descritores + Scatterplots
# 
# Este arquivo contém TODO o código necessário para:
# 1. Processar TODAS as imagens do dataset sagital
# 2. Calcular 6 descritores morfológicos para cada imagem
# 3. Gerar scatterplots coloridos por classe (15 gráficos únicos)
# ===============================================================================

# ------------------- Imports ---------------------

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from scipy.spatial.distance import pdist
import pandas as pd
import glob
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# ===============================================================================
# CLASSE 1: SEGMENTAÇÃO DE VENTRÍCULOS
# ===============================================================================

class VentricleSegmentation:
    """
    Classe responsável pela segmentação dos ventrículos laterais em imagens de ressonância magnética.
    Utiliza técnicas de crescimento de regiões com seleção automática de seed.
    """

    # Construtor do objeto que vai fazer as segmentações
    # Tem como váriaveis:
    # image_path -> caminho da imagem (formato NIfTI .nii.gz)
    # img_data -> dados da imagem após carregamento
    # slice_idx -> índice do corte selecionado (para imagens 3D)
    # seed -> posição (y, x) da seed selecionada para crescimento de regiões
    # brain_mask -> máscara binária que separa cérebro do fundo
    def __init__(self, image_path):
        """
        Inicializa o segmentador de ventrículos laterais.
        
        Args:
            image_path: Caminho para o arquivo NIfTI (.nii.gz)
        """
        self.image_path = Path(image_path)
        self.img_data = None
        self.slice_idx = None
        self.seed = None
        self.brain_mask = None


    # Preparar a imagem pra uso (segmentar e etc)
    # Carrega a imagem e guarda ela em nii_img e depois em img_data guarda as informações da imagem
    # Printa o img_data.shape, que nada mais é do que o tamanho dela (altura x largura)
    # Verifica se a imagem é 2D ou 3D e se for 3D transforma ela em imagem sagital 
    # (no caso todas as nossas imagens são sagitais, então essa parte de transformar de 3D pra 2D não serve pra nada)
    def load_image(self):
        """Carrega a imagem NIfTI e seleciona o corte sagital central."""
        nii_img = nib.load(self.image_path)
        self.img_data = nii_img.get_fdata()
        
        # Verificar dimensionalidade da imagem
        if self.img_data.ndim == 2:
            # Imagem já é 2D
            return self.img_data
        elif self.img_data.ndim == 3:
            # Imagem 3D - seleciona corte sagital central
            self.slice_idx = self.img_data.shape[0] // 2
            return self.img_data[self.slice_idx, :, :]
        else:
            raise ValueError(f"Dimensionalidade não suportada: {self.img_data.ndim}D")


    # Função que normaliza a imagem (transforma os valores para o intervalo [0, 255])
    # Pega o valor mínimo da imagem e subtrai de todos os pixels
    # Depois divide pelo valor máximo e multiplica por 255
    # Resultado: imagem com valores entre 0 e 255 (formato padrão de visualização)
    def normalize_image(self, img):
        """Normaliza a imagem para range [0, 255]."""
        img_norm = img - np.min(img)
        img_norm = img_norm / (np.max(img_norm) + 1e-8) * 255
        return img_norm.astype(np.uint8)


    # Como o fundo tem a mesma cor do Ventrículo lateral, se faz necessário analisar só o cérebro,
    # pq se não fica difícil de selecionar a seed corretamente. Essa função tem esse papel de separar
    # o cérebro do fundo pra depois a gente escolher a seed.
    # 
    # Faz o processo por segmentação por limiarização utilizando Otsu pra encontrar a melhor 
    # separação entre o objeto (cérebro) e o fundo que queremos remover.
    def create_brain_mask(self, img):
        """
        Cria uma máscara do cérebro para excluir o fundo.
        
        Args:
            img: Imagem 2D normalizada
            
        Returns:
            Máscara binária do cérebro
        """

        # Usar threshold de Otsu para separar cérebro do fundo
        # Faz Otsu bem automático mesmo
        threshold = filters.threshold_otsu(img)
        

        # Os ventrículos têm tons de cinza escuros como o fundo, porém o fundo é mais escuro.
        # O ponto de usar apenas metade do threshold é excluir apenas os pontos mais escuros da imagem,
        # pois se usássemos o threshold sem essa multiplicação por 0.5 existe uma chance do ventrículo
        # ser considerado como fundo ao invés de parte do objeto, o que não é o que queremos.
        brain_mask = img > threshold * 0.5
        
        # Morfologia matemática pra tirar os ruídos e corrigir uns BO da imagem pós limiarização.
        # Faz nesse caso a abertura, depois o fechamento e depois tampa buraco
        # (necessário ver depois o por que de fazer nessa ordem de abertura e fechamento 
        # e por que usar elemento estruturante de 5x5 e 11x11)
        brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((5, 5)))
        brain_mask = ndimage.binary_closing(brain_mask, structure=np.ones((11, 11)))
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        
        # Remover pequenas componentes (ruído).
        # Faz todo um processo numerando pequenos objetos que ficaram separados do objeto principal (cérebro)
        # e marca todos os objetos da imagem com labels (inclusive o cérebro).
        # Passa por todos os labels verificando e guardando em um vetor o tamanho deles em pixels
        # e depois pega o label que tiver mais pixels (objeto principal vulgo cérebro).
        # Depois coloca na máscara (brain_mask) apenas os componentes que têm labels iguais ao label
        # de maior quantidade de pixels
        labeled, num_features = ndimage.label(brain_mask)
        if num_features > 0:
            sizes = ndimage.sum(brain_mask, labeled, range(1, num_features + 1))
            # Manter apenas a maior componente (o cérebro)
            max_label = np.argmax(sizes) + 1
            brain_mask = labeled == max_label
        
        # Faz uma erosão pra garantir que tá pegando dentro do cérebro mesmo
        brain_mask = ndimage.binary_erosion(brain_mask, structure=np.ones((15, 15)))
        
        # Retorna a máscara que define o que é cérebro na imagem
        return brain_mask


    # Função que vai gerar várias seeds e vai no final escolher qual é a melhor seed para definir
    # o ventrículo dentre as seeds criadas.
    # 
    # Cria as seeds dentro da região do cérebro (já excluímos o fundo) colocando as seeds em regiões
    # mais escuras (tons de cinza mais escuros). Nesse caso tá gerando 50 seeds pra escolher uma só.
    def generate_candidate_seeds(self, img, brain_mask, n_candidates=50):
        """
        Gera candidatos a seed baseado em características dos ventrículos.
        Ventrículos aparecem como regiões escuras (baixa intensidade) DENTRO do cérebro.
        
        Args:
            img: Imagem 2D normalizada
            brain_mask: Máscara binária do cérebro
            n_candidates: Número de candidatos a gerar
            
        Returns:
            Lista de candidatos (y, x) com suas características
        """
        # Coloca a altura e largura da imagem nas variáveis h e w
        h, w = img.shape
        # Cria um array vazio de candidatos a seed
        candidates = []
        
        # Cria uma cópia da imagem e nessa cópia faz tudo que não faz parte do cérebro (definido na 
        # brain_mask) ganhar o valor 255 (referente ao branco). Como um dos parâmetros para definir
        # a seed é o quão escura é a região, o fundo não será mais escolhido, uma vez que é 
        # completamente branco
        img_masked = img.copy()
        img_masked[~brain_mask] = 255  # Fundo vira branco (alta intensidade)
        
        # Aplicar threshold para encontrar regiões escuras DENTRO do cérebro
        # Ventrículos são as regiões MAIS ESCURAS dentro do cérebro
        # Retorna um threshold no qual 15% dos meus dados estão abaixo (no caso são mais escuros)
        dark_threshold = np.percentile(img_masked[brain_mask], 15)
        # Define as regiões escuras do cérebro pegando apenas regiões menores que o threshold
        # (dark_threshold) e que fazem parte da brain_mask
        dark_regions = (img_masked < dark_threshold) & brain_mask
        
        # Aplicar morfologia para limpar ruído (no caso faz abertura depois fechamento).
        # Se faz necessário entender por que ele está fazendo essas operações nessa ordem
        # e com os elementos estruturantes 3x3 e 5x5
        dark_regions = ndimage.binary_opening(dark_regions, structure=np.ones((3,3)))
        dark_regions = ndimage.binary_closing(dark_regions, structure=np.ones((5,5)))
        
        # Calcular distância das bordas das regiões escuras
        # O que ele faz é retornar a distância euclidiana de todos os pontos classificados como
        # objeto em relação ao fundo
        distance_map = ndimage.distance_transform_edt(dark_regions)
        
        # Encontrar máximos locais na distância (centro das regiões escuras) e define os candidatos
        # a seed verificando se a distância deles até o fundo for mais que 3 pixels
        local_max = filters.rank.maximum(distance_map.astype(np.uint8), np.ones((15, 15)))
        seed_candidates = (distance_map == local_max) & (distance_map > 3)
        
        # Obter coordenadas dos candidatos
        y_coords, x_coords = np.where(seed_candidates)
        
        for y, x in zip(y_coords, x_coords):
            # VERIFICAÇÃO CRÍTICA: Garantir que está dentro da máscara do cérebro
            if not brain_mask[y, x]:
                continue
            
            # Definir região de análise ao redor do candidato
            # Delimita onde começa e onde termina a janela de análise garantindo que vai estar
            # dentro da imagem. Tamanho da janela é 15 pra todos os lados em relação ao ponto candidato
            window_size = 15
            y_min = max(0, y - window_size)
            y_max = min(h, y + window_size)
            x_min = max(0, x - window_size)
            x_max = min(w, x + window_size)
            
            # Pega a janela que vai ser analisada tanto na imagem quanto na máscara do cérebro
            # (pra fazer a análise em ambos e garantir que no final a parte da janela que estamos
            # analisando está dentro do cérebro de fato)
            window = img[y_min:y_max, x_min:x_max]
            window_mask = brain_mask[y_min:y_max, x_min:x_max]
            
            # Calcular características apenas na região do cérebro
            if np.sum(window_mask) < 50:  # Região muito pequena
                continue
            
            # Coloca na análise da janela apenas as partes que estão dentro do cérebro
            window_values = window[window_mask]
            
            # Calcular características de cada candidato a seed
            features = {
                'position': (y, x),  # Posição (y, x) da seed
                'intensity': float(img[y, x]),  # O quão claro é a seed (0-255)
                'mean_intensity': float(np.mean(window_values)),  # Intensidade média da região ao redor
                'std_intensity': float(np.std(window_values)),  # Desvio padrão da janela
                'homogeneity': float(1.0 / (1.0 + np.std(window_values))),  # Define se é homogêneo ou heterogêneo usando o DP
                'centrality': 1.0 - np.sqrt(((y - h/2)/h)**2 + ((x - w/2)/w)**2),  # Posição relativa (ventrículos geralmente na região central do cérebro)
                'distance_from_edge': float(distance_map[y, x]),  # Distância em relação ao fim da região escura
                'distance_from_brain_edge': float(ndimage.distance_transform_edt(brain_mask)[y, x])  # Distância da borda do cérebro (ventrículos estão no interior)
            }
            
            # Coloca num array as características de cada seed que foi encontrada.
            # Essas características vão ser analisadas em outra função pra definir a melhor seed
            candidates.append(features)
        
        # Retorna a lista de seeds candidatas (limitada a n_candidates)
        return candidates[:n_candidates]


    # Função que vai ponderar e dar notas para cada uma das seeds.
    # A que tiver melhor nota depois provavelmente vai ser escolhida como seed do ventrículo.
    # Mas essa função só dá as notas, quem escolhe é outra.
    # 
    # Ventrículos laterais têm as seguintes características:
    # - Escuros (líquido cefalorraquidiano)
    # - Homogêneos (região uniforme)
    # - Centrais (próximos ao centro do cérebro)
    # - Interiores (longe das bordas do cérebro)
    def score_seed(self, features):
        """
        Calcula um score para cada candidato baseado em suas características.
        Ventrículos laterais: escuros, homogêneos, centrais, dentro do cérebro.
        
        Args:
            features: Dicionário com características do candidato
            
        Returns:
            Score do candidato (maior = melhor)
        """
        score = 0.0
        
        # Baixa intensidade (ventrículos são escuros - líquido cefalorraquidiano)
        # Se a intensidade é alta (muito perto do branco) intensity_score fica mais próximo de 0.
        # Se intensidade muito baixa (muito perto do preto) intensity_score fica mais próximo de 1.
        # Soma o intensity_score ao score total (com peso de 3.5)
        intensity_score = 1.0 - (features['intensity'] / 255.0) 
        score += intensity_score * 3.5
        
        # Homogeneidade (região uniforme)
        # Verifica se a região é coerente com a cor da seed, isto é, não tem muita variação de
        # cores nas seeds da região (DP baixo). Soma a homogeneidade ao score total (com peso de 2.5)
        score += features['homogeneity'] * 2.5
        
        # Centralidade (ventrículos próximos ao centro)
        # Verifica se a região está próxima do centro do cérebro (o resultado está entre 0 e 1).
        # Soma a centralidade ao score total (com peso de 2.0)
        score += features['centrality'] * 2.0
        
        # Distância da borda da região escura
        # Verifica o quão longe a seed está do fim da região escura à qual ela pertence.
        # Se a distância é maior que 20 pixels até a borda da região escura, o score será de 1.5.
        # Se for menor do que 20 pixels, será esse valor/20 * 1.5 (que é menor sempre do que 1.5)
        score += min(features['distance_from_edge'] / 20.0, 1.0) * 1.5
        
        # Distância da borda do cérebro
        # Se a distância for maior que 30, recebe score 4.
        # Se tiver distância menor que 30 pixels, recebe distância/30 * 4
        # (obrigatoriamente menor, logo o critério foi penalizado)
        score += min(features['distance_from_brain_edge'] / 30.0, 1.0) * 4.0
        
        # Retorna o Score total de cada seed candidata
        return score
    

    # Selecionar a melhor seed com base nos scores calculados para cada seed candidata.
    # Retorna o que foi considerada a melhor seed (ventrículo lateral).
    # Essa função é quem chama a função de gerar o score para cada seed.
    def select_best_seed(self, candidates):
        """
        Seleciona a melhor seed entre os candidatos.
        
        Args:
            candidates: Lista de candidatos com características
            
        Returns:
            Coordenadas (y, x) da melhor seed
        """
        if not candidates:
            # Se não encontrar candidatos, retorna None
            return None
        
        # Calcular scores
        # Criar um array de scores e seeds (Score, posição da seed) com os resultados de cada candidato
        scored_candidates = [(self.score_seed(c), c['position']) for c in candidates]
        # Ordena os scores em ordem decrescente (como é uma tupla tem que selecionar qual parte
        # da tupla escolher pra ordenar, no caso a de índice 0, vulgo o score)
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Pega o elemento com index 0 da lista (O de melhor score) e retorna o segundo valor
        # da tupla [1], no caso a posição da seed
        return scored_candidates[0][1]
    

    # Essa função detém o algoritmo de crescimento de regiões.
    # Recebe a imagem, a seed e o threshold (limite de diferença de intensidade aceito).
    # 
    # O algoritmo funciona assim:
    # 1. Começa pela seed
    # 2. Olha os 8 vizinhos ao redor
    # 3. Se o vizinho tem cor parecida (diferença < threshold), inclui no ventrículo
    # 4. Repete o processo para cada novo pixel incluído
    # 5. Para quando não houver mais pixels similares
    def region_growing(self, img, seed, threshold=20):
        """
        Algoritmo de crescimento de regiões.
        
        Args:
            img: Imagem 2D
            seed: Tupla (y, x) com posição inicial
            threshold: Limiar de diferença de intensidade
            
        Returns:
            Máscara binária da região segmentada
        """

        # Guarda nas variáveis h e w a altura e largura da imagem
        h, w = img.shape
        # Cria uma matriz para guardar o resultado com a mesma altura e mesma largura da imagem
        # original sendo de tipo booleano (já que vai ser só objeto e fundo)
        segmented = np.zeros((h, w), dtype=bool)
        
        # Se não tiver seed, retorna a matriz vazia (não tem ventrículo pra chamar de objeto)
        if seed is None:
            return segmented
        
        # Pega o valor do pixel da seed (o tom de cinza dela entre 0 e 255)
        seed_value = img[seed[0], seed[1]]
        
        # Cria uma fila para processar pixels
        queue = [seed]
        # Marca a seed como parte do ventrículo
        segmented[seed[0], seed[1]] = True
        
        # 8-conectividade (pixels ao redor, vizinhos ou derivados disso)
        # Representa os 8 vizinhos: cima-esquerda, cima, cima-direita, esquerda, direita,
        # baixo-esquerda, baixo, baixo-direita
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Enquanto ainda houverem pixels a serem analisados
        while queue:
            # Remove o pixel da fila e guarda suas coordenadas em x e y
            y, x = queue.pop(0)
            
            # Para cada vizinho (-1,-1) (-1,0) (-1,1) etc...
            for dy, dx in neighbors:
                # Pega a posição real do vizinho (coordenada do pixel + vizinho)
                ny, nx = y + dy, x + dx
                
                # Verificar limites (se tá dentro da imagem e não faz parte do ventrículo ainda...)
                if 0 <= ny < h and 0 <= nx < w and not segmented[ny, nx]:
                    # CRÍTICO: Verificar se está dentro da máscara do cérebro
                    # Se a máscara do cérebro existir mas o pixel em questão não está dentro
                    # do cérebro, pula para o próximo vizinho
                    if self.brain_mask is not None and not self.brain_mask[ny, nx]:
                        continue
                    
                    # Se estiver dentro do cérebro, verificar similaridade
                    # Se o pixel analisado tiver diferença em tons de cinza menor que o threshold
                    # (ainda é considerado objeto)
                    if abs(float(img[ny, nx]) - float(seed_value)) < threshold:
                        # O pixel vira Ventrículo
                        segmented[ny, nx] = True
                        # O pixel entra na Fila para encontrar seus vizinhos
                        queue.append((ny, nx))
        
        # Pós-processamento: remover pequenas regiões e preencher buracos
        # Depois de fazer tudo isso e definir o ventrículo, pega o resultado e faz uma abertura
        # seguida de fechamento e tapar buracos (se faz necessário descobrir e entender o por que
        # fazer abertura e depois fechamento com os objetos estruturantes 3x3 e 5x5)
        segmented = ndimage.binary_opening(segmented, structure=np.ones((3,3)))
        segmented = ndimage.binary_closing(segmented, structure=np.ones((5,5)))
        segmented = ndimage.binary_fill_holes(segmented)
        
        return segmented


    # Função que é chamada para fazer todo o processo de segmentação de uma vez só.
    # Executa todas as etapas: carregamento, normalização, criação da máscara do cérebro,
    # geração de seeds, seleção da melhor seed e crescimento de regiões.
    # 
    # Retorna a máscara binária da segmentação (True = ventrículo, False = fundo) e a imagem normalizada
    def process_segmentation(self, threshold=20, verbose=False):
        """
        Executa todo o pipeline de segmentação.
        
        Args:
            threshold: Limiar para crescimento de regiões
            verbose: Se True, mostra informações detalhadas do processamento
            
        Returns:
            Tupla (segmentation, img_normalized)
        """
        if verbose:
            print(f"\n[Segmentando] {self.image_path.name}")
        
        # Carregar a imagem por meio de nossa função load_image
        img = self.load_image()
        # Normalizar a imagem para o intervalo [0, 255]
        img_norm = self.normalize_image(img)
        
        # Criar a máscara do cérebro chamando nossa função passando como parâmetro a imagem já normalizada
        self.brain_mask = self.create_brain_mask(img_norm)
        
        # Gerar candidatos a seed, chamando nossa função, passando como parâmetro
        # a imagem já normalizada e a máscara do cérebro
        candidates = self.generate_candidate_seeds(img_norm, self.brain_mask)
        
        # Selecionar a melhor seed, que será usada para realizar o crescimento de regiões,
        # por meio da nossa função de seleção de seed
        self.seed = self.select_best_seed(candidates)
        
        # Se não tiver encontrado nenhuma seed usa o centro do cérebro como seed
        if self.seed is None and self.brain_mask is not None:
            # Encontrar centro de massa do cérebro
            y_coords, x_coords = np.where(self.brain_mask)
            if len(y_coords) > 0:
                self.seed = (int(np.mean(y_coords)), int(np.mean(x_coords)))
        
        # Executar o crescimento de regiões passando como parâmetro a imagem normalizada,
        # a seed encontrada, e um threshold pré-definido
        segmentation = self.region_growing(img_norm, self.seed, threshold)
        
        if verbose:
            # Número de pixels no ventrículo
            seg_pixels = np.sum(segmentation)
            print(f"  ✓ Pixels segmentados: {seg_pixels}")
        
        # Retorna a segmentação (máscara binária) e a imagem normalizada
        return segmentation, img_norm


# ===============================================================================
# CLASSE 2: EXTRAÇÃO DE DESCRITORES MORFOLÓGICOS
# ===============================================================================

class VentricleDescriptors:
    """
    Classe responsável pelo cálculo de descritores morfológicos dos ventrículos laterais.
    Extrai 6 características que descrevem a forma e tamanho dos ventrículos segmentados.
    Baseado nos slides de Representação e Descrição (Prof. Alexei Machado).
    """
    
    def __init__(self):
        """
        Construtor do objeto que vai calcular os descritores.
        Inicializa um dicionário vazio que vai armazenar os 6 descritores calculados.
        """
        self.descriptors = {}
    
    def calculate_all_descriptors(self, binary_mask, verbose=False):
        """
        Calcula todos os 6 descritores para uma máscara binária dos ventrículos.
        Esta é a função principal que chama todas as outras funções de cálculo.
        
        Recebe:
        - binary_mask: Imagem binária (True/False ou 0/255) com os ventrículos segmentados
        - verbose: Se True, mostra mensagens de debug durante o cálculo
        
        Retorna:
        - Dicionário com os 6 descritores calculados
        """
        # Garantir que a máscara é binária no formato correto (0 ou 255)
        # Se os valores forem booleanos (True/False), converte para 0/255
        if binary_mask.dtype == bool:
            binary_mask = (binary_mask.astype(np.uint8)) * 255
        elif binary_mask.max() <= 1:
            binary_mask = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask = binary_mask.astype(np.uint8)
        
        # Encontrar contornos usando OpenCV
        # RETR_EXTERNAL: pega apenas o contorno externo (ignora buracos internos)
        # CHAIN_APPROX_SIMPLE: comprime segmentos horizontais, verticais e diagonais
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Se não encontrou nenhum contorno (ventrículo não detectado), retorna valores vazios
        if len(contours) == 0:
            return self._empty_descriptors()
        
        # Usar o maior contorno (assumindo que é o ventrículo principal)
        # Isso garante que pegamos o ventrículo e não algum ruído pequeno
        main_contour = max(contours, key=cv2.contourArea)
        
        # ===== CALCULAR OS 3 DESCRITORES OBRIGATÓRIOS =====
        
        # 1. Área: quantidade total de pixels do ventrículo
        area = self._calculate_area(main_contour)
        
        # Perímetro: necessário para calcular a circularidade
        # True = contorno fechado
        perimeter = cv2.arcLength(main_contour, True)
        
        # 2. Circularidade: o quão próximo o ventrículo está de um círculo perfeito
        circularity = self._calculate_circularity(area, perimeter)
        
        # 3. Excentricidade: o quão alongado é o ventrículo
        eccentricity = self._calculate_eccentricity(main_contour)
        
        # ===== CALCULAR OS 3 DESCRITORES ADICIONAIS =====
        
        # 4. Retangularidade: quão bem o ventrículo preenche seu retângulo envolvente
        rectangularity = self._calculate_rectangularity(main_contour, area)
        
        # 5. Solidez: presença de concavidades nas bordas do ventrículo
        solidity = self._calculate_solidity(main_contour, area)
        
        # 6. Diâmetro: maior distância entre dois pontos do ventrículo
        diameter = self._calculate_diameter(main_contour)
        
        # Guardar todos os descritores em um dicionário
        self.descriptors = {
            # Obrigatórios
            'area': area,
            'circularity': circularity,
            'eccentricity': eccentricity,
            # Adicionais
            'rectangularity': rectangularity,
            'solidity': solidity,
            'diameter': diameter
        }
        
        # Retorna o dicionário com todos os descritores
        return self.descriptors
    
    # ========== FUNÇÕES PARA CALCULAR OS DESCRITORES OBRIGATÓRIOS ==========
    
    def _calculate_area(self, contour):
        """
        Calcula a ÁREA da região em pixels (Slide 13).
        
        Área = quantidade total de pixels dentro do contorno do ventrículo.
        Quanto maior a área, maior o volume do ventrículo.
        Em pacientes com Alzheimer, os ventrículos tendem a aumentar com a progressão da doença.
        
        Recebe:
        - contour: contorno do ventrículo extraído pelo OpenCV
        
        Retorna:
        - Área em pixels (valor numérico float)
        """
        return cv2.contourArea(contour)
    
    def _calculate_circularity(self, area, perimeter):
        """
        Calcula a CIRCULARIDADE/COMPACIDADE: C = 4πA / P² (Slide 13).
        
        Mede o quão próximo o ventrículo está de um círculo perfeito.
        - Valor = 1.0: círculo perfeito
        - Valor < 1.0: forma irregular (quanto menor, mais irregular)
        
        Ventrículos com Alzheimer tendem a ter formas mais irregulares (circularidade menor).
        
        Recebe:
        - area: área do ventrículo em pixels
        - perimeter: perímetro do ventrículo em pixels
        
        Retorna:
        - Circularidade (valor entre 0 e 1)
        """
        # Se o perímetro for zero (caso impossível, mas por segurança), retorna 0
        if perimeter == 0:
            return 0
        
        # Aplica a fórmula: 4 * pi * área / perímetro²
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return circularity
    
    def _calculate_eccentricity(self, contour):
        """
        Calcula a EXCENTRICIDADE: Eixo máximo / Eixo mínimo (Slide 14).
        
        Mede o quanto o ventrículo é alongado:
        - Valor próximo a 1: forma circular/quadrada
        - Valor > 1: forma alongada (quanto maior, mais alongado)
        
        Para calcular, ajusta uma elipse no contorno e pega a razão entre seus eixos.
        Ventrículos podem se expandir de forma assimétrica no Alzheimer.
        
        Recebe:
        - contour: contorno do ventrículo
        
        Retorna:
        - Excentricidade (valor >= 1)
        """
        # Para ajustar uma elipse, precisamos de pelo menos 5 pontos
        if len(contour) < 5:
            # Se tiver poucos pontos, usa o retângulo envolvente como alternativa
            x, y, w, h = cv2.boundingRect(contour)
            # Evita divisão por zero
            if min(w, h) == 0:
                return 1
            # Retorna a razão entre largura e altura
            return max(w, h) / min(w, h)
        
        try:
            # Ajusta uma elipse no contorno
            # ellipse = ((centro_x, centro_y), (eixo_maior, eixo_menor), ângulo)
            ellipse = cv2.fitEllipse(contour)
            # Pega o maior e o menor eixo da elipse
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            
            # Evita divisão por zero
            if minor_axis == 0:
                return 1
            
            # Calcula a excentricidade como a razão entre os eixos
            eccentricity = major_axis / minor_axis
            return eccentricity
        except:
            # Se houver erro ao ajustar a elipse, retorna 1 (forma circular)
            return 1
    
    # ========== FUNÇÕES PARA CALCULAR OS DESCRITORES ADICIONAIS ==========
    
    def _calculate_rectangularity(self, contour, area):
        """
        Calcula a RETANGULARIDADE: Área / Área do Retângulo Básico (Slide 14).
        
        Retângulo Básico = menor retângulo que envolve completamente o ventrículo.
        Mede quão bem o ventrículo preenche seu retângulo envolvente:
        - Valor = 1.0: preenche completamente o retângulo (ex: um retângulo perfeito)
        - Valor < 1.0: tem espaços vazios (quanto menor, mais irregular)
        
        Útil para identificar protuberâncias e irregularidades na forma do ventrículo.
        
        Recebe:
        - contour: contorno do ventrículo
        - area: área do ventrículo em pixels
        
        Retorna:
        - Retangularidade (valor entre 0 e 1)
        """
        # Obtém o retângulo mínimo (pode estar rotacionado para ser o menor possível)
        # rect = ((centro_x, centro_y), (largura, altura), ângulo)
        rect = cv2.minAreaRect(contour)
        # Pega a largura e altura do retângulo
        width, height = rect[1]
        # Calcula a área do retângulo
        rect_area = width * height
        
        # Evita divisão por zero
        if rect_area == 0:
            return 0
        
        # Calcula a retangularidade: área do ventrículo / área do retângulo
        rectangularity = area / rect_area
        return rectangularity
    
    def _calculate_solidity(self, contour, area):
        """
        Calcula a SOLIDEZ: Área / Área do Fecho Convexo (Slide 19).
        
        Fecho Convexo = menor polígono convexo que engloba todo o ventrículo.
        Mede o quão "preenchida" é a forma:
        - Valor = 1.0: forma completamente convexa (sem concavidades)
        - Valor < 1.0: tem concavidades nas bordas (quanto menor, mais concavidades)
        
        A diferença (Fecho Convexo - Área) é chamada de "deficiência convexa".
        Concavidades irregulares podem indicar atrofia cerebral irregular no Alzheimer.
        
        Recebe:
        - contour: contorno do ventrículo
        - area: área do ventrículo em pixels
        
        Retorna:
        - Solidez (valor entre 0 e 1)
        """
        # Calcula o fecho convexo (convex hull) do contorno
        # O fecho convexo é como se colocássemos um elástico ao redor do ventrículo
        hull = cv2.convexHull(contour)
        # Calcula a área do fecho convexo
        hull_area = cv2.contourArea(hull)
        
        # Evita divisão por zero
        if hull_area == 0:
            return 0
        
        # Calcula a solidez: área do ventrículo / área do fecho convexo
        solidity = area / hull_area
        return solidity
    
    def _calculate_diameter(self, contour):
        """
        Calcula o DIÂMETRO: Maior distância entre dois pontos do contorno (Slide 13).
        
        Representa a maior dimensão linear do ventrículo.
        É útil para detectar expansão ventricular, comum no Alzheimer.
        
        O cálculo é feito medindo a distância euclidiana entre todos os pares de pontos
        do contorno e pegando a maior distância encontrada.
        
        Para contornos muito grandes (>100 pontos), fazemos uma amostragem para otimizar
        o processamento (senão ficaria muito lento).
        
        Recebe:
        - contour: contorno do ventrículo
        
        Retorna:
        - Diâmetro em pixels (valor numérico float)
        """
        # Converter contorno para array de pontos (remover dimensão extra)
        # Contorno vem no formato (n, 1, 2), queremos (n, 2)
        points = contour.reshape(-1, 2)
        
        # Se tiver menos de 2 pontos, não dá pra calcular distância
        if len(points) < 2:
            return 0
        
        # Se o contorno tiver muitos pontos (>100), faz amostragem para otimizar
        # Calcular distância entre todos os pares de 1000 pontos seria muito lento
        if len(points) > 100:
            # Seleciona 100 pontos igualmente espaçados ao longo do contorno
            indices = np.linspace(0, len(points)-1, 100, dtype=int)
            points = points[indices]
        
        # Calcula a matriz de distâncias entre todos os pares de pontos
        # pdist retorna um vetor com todas as distâncias par a par
        distances = pdist(points, metric='euclidean')
        # Pega a maior distância encontrada (esse é o diâmetro)
        diameter = np.max(distances)
        
        return diameter
    
    # ========== FUNÇÕES UTILITÁRIAS ==========
    
    def _empty_descriptors(self):
        """
        Retorna descritores vazios (todos zeros) quando não há contorno detectado.
        Isso acontece quando a segmentação falha ou o ventrículo não foi encontrado.
        """
        return {
            'area': 0,
            'circularity': 0,
            'eccentricity': 0,
            'rectangularity': 0,
            'solidity': 0,
            'diameter': 0
        }


# ===============================================================================
# FUNÇÃO: PROCESSAR DATASET COMPLETO
# ===============================================================================

def process_full_dataset(images_dir, demographic_csv, output_csv, threshold=20):
    """
    Processa TODAS as imagens do dataset e gera CSV com descritores.
    
    Esta função:
    1. Busca todas as imagens .nii.gz no diretório especificado
    2. Para cada imagem, executa a segmentação dos ventrículos
    3. Calcula os 6 descritores morfológicos
    4. Faz merge com os dados demográficos para incluir a classe (Group)
    5. Salva tudo em um CSV único
    
    Recebe:
    - images_dir: diretório contendo as imagens NIfTI (.nii.gz)
    - demographic_csv: caminho do CSV com dados demográficos (incluindo Group)
    - output_csv: caminho do arquivo CSV de saída
    - threshold: limiar para crescimento de regiões (padrão: 20)
    
    Retorna:
    - DataFrame com todos os descritores e classes calculados
    """
    print("\n" + "=" * 80)
    print("PROCESSAMENTO DO DATASET COMPLETO")
    print("=" * 80)
    
    # Buscar todas as imagens .nii.gz no diretório
    image_paths = sorted(glob.glob(str(Path(images_dir) / "*.nii.gz")))
    
    if len(image_paths) == 0:
        print(f"\n❌ ERRO: Nenhuma imagem encontrada em {images_dir}")
        print("   Verifique se o caminho está correto e se contém arquivos .nii.gz")
        return None
    
    print(f"\n✓ Encontradas {len(image_paths)} imagens para processar")
    print(f"✓ Diretório: {images_dir}")
    print(f"✓ Threshold: {threshold}\n")
    
    # Carregar dados demográficos
    print("Carregando dados demográficos...")
    df_demographic = pd.read_csv(demographic_csv, delimiter=';')
    print(f"✓ Dados demográficos carregados: {len(df_demographic)} registros\n")
    
    # Lista para armazenar resultados
    all_results = []
    
    # Processar cada imagem
    for idx, img_path in enumerate(image_paths, 1):
        try:
            # Extrair o MRI ID do nome do arquivo (ex: OAS2_0001_MR1)
            # Remove sufixos como '_sag' e extensões '.nii'
            image_name = Path(img_path).stem.replace('_sag', '').replace('.nii', '')
            
            print(f"[{idx}/{len(image_paths)}] Processando: {image_name}")
            
            # ===== ETAPA 1: SEGMENTAR =====
            # Criar o objeto segmentador passando o caminho da imagem
            segmenter = VentricleSegmentation(img_path)
            # Executar a segmentação completa
            segmentation, img_normalized = segmenter.process_segmentation(
                threshold=threshold,
                verbose=False
            )
            
            # ===== ETAPA 2: CALCULAR DESCRITORES =====
            # Criar o objeto extrator de descritores
            descriptor_extractor = VentricleDescriptors()
            # Calcular todos os 6 descritores morfológicos
            descriptors = descriptor_extractor.calculate_all_descriptors(segmentation, verbose=False)
            
            # ===== ETAPA 3: ADICIONAR INFORMAÇÕES =====
            # Adicionar o identificador da imagem (MRI ID)
            descriptors['MRI ID'] = image_name
            
            # Buscar a classe (Group) nos dados demográficos
            # Filtra o DataFrame demográfico pelo MRI ID atual
            group_info = df_demographic[df_demographic['MRI ID'] == image_name]
            if len(group_info) > 0:
                # Se encontrou informação, pega o valor da coluna 'Group'


                #AQUI GUSTAVO !!! 
                # Se Grupo for converted, muda para Nondemented se CRD = 0 e para Demented se CRD > 0
                if group_info['Group'].values[0] == 'Converted':
                    if group_info['CDR'].values[0] == 0:
                        descriptors['Group'] = 'Nondemented'
                    else:
                        descriptors['Group'] = 'Demented'
                else:   
                    descriptors['Group'] = group_info['Group'].values[0]
            else:
                # Se não encontrou, marca como 'Unknown'
                descriptors['Group'] = 'Unknown'
            
            # Guardar resultados na lista
            all_results.append(descriptors)
            
            # Feedback de progresso para o usuário
            if descriptors['area'] > 0:
                print(f"  ✓ Área: {descriptors['area']:.0f} px | Grupo: {descriptors['Group']}")
            else:
                print(f"  ⚠ Ventrículo não detectado | Grupo: {descriptors['Group']}")
            
        except Exception as e:
            print(f"  ❌ ERRO ao processar {img_path}: {e}")
            continue
    
    # ===== CRIAR DATAFRAME COM RESULTADOS =====
    print(f"\n{'='*80}")
    print("CRIANDO DATAFRAME COM RESULTADOS")
    print(f"{'='*80}\n")
    
    # Converte a lista de dicionários em DataFrame do pandas
    df = pd.DataFrame(all_results)
    
    # Reorganizar colunas para melhor visualização
    # Coloca MRI ID e Group primeiro, depois os descritores
    cols_order = ['MRI ID', 'Group', 'area', 'circularity', 'eccentricity', 
                  'rectangularity', 'solidity', 'diameter']
    df = df[cols_order]
    
    # Salvar CSV
    df.to_csv(output_csv, index=False)
    
    print(f"✓ CSV gerado: {output_csv}")
    print(f"✓ Total de imagens processadas: {len(df)}")
    print(f"✓ Imagens com ventrículo detectado: {(df['area'] > 0).sum()}")
    print(f"✓ Imagens sem detecção: {(df['area'] == 0).sum()}")
    
    # Mostrar distribuição por classe
    print(f"\n✓ Distribuição por classe:")
    for group in df['Group'].unique():
        count = (df['Group'] == group).sum()
        print(f"   • {group}: {count} imagens")
    
    # ===== MOSTRAR ESTATÍSTICAS RESUMIDAS =====
    print("\n" + "=" * 80)
    print("ESTATÍSTICAS DOS DESCRITORES")
    print("=" * 80)
    
    # Lista com os nomes das colunas dos descritores
    descriptor_cols = ['area', 'circularity', 'eccentricity', 
                       'rectangularity', 'solidity', 'diameter']
    
    # Filtrar apenas imagens com ventrículo detectado para estatísticas
    # (área > 0 significa que encontrou o ventrículo)
    df_valid = df[df['area'] > 0]
    
    # Para cada descritor, calcular e mostrar estatísticas
    for col in descriptor_cols:
        print(f"\n{col.upper()}:")
        print(f"  Média:  {df_valid[col].mean():.4f}")
        print(f"  Desvio: {df_valid[col].std():.4f}")
        print(f"  Mín:    {df_valid[col].min():.4f}")
        print(f"  Máx:    {df_valid[col].max():.4f}")
    
    print("\n" + "=" * 80)
    
    return df


# ===============================================================================
# FUNÇÃO: GERAR SCATTERPLOTS (ITEM 8)
# ===============================================================================

def generate_scatterplots(descriptors_csv, output_dir='scatterplots'):
    """
    Gera scatterplots para todas as combinações ÚNICAS de descritores (item 8).
    
    Esta função:
    1. Lê o CSV com os descritores e classes
    2. Gera 15 gráficos (combinações únicas de 6 características: C(6,2) = 15)
    3. Colore os pontos de acordo com a classe (Group)
    4. Salva todos os gráficos no diretório especificado
    
    Plota 15 gráficos únicos (não repete invertendo X e Y)
    Cores conforme especificação do trabalho:
    - Preto = Converted
    - Azul = Nondemented
    - Vermelho = Demented
    
    Recebe:
    - descriptors_csv: CSV com os descritores calculados e a coluna 'Group'
    - output_dir: diretório para salvar os scatterplots (padrão: 'scatterplots')
    
    Retorna:
    - None (salva os gráficos em arquivos PNG)
    """
    print("\n" + "=" * 80)
    print("GERANDO SCATTERPLOTS (ITEM 8)")
    print("=" * 80)
    
    # ===== CRIAR DIRETÓRIO DE SAÍDA =====
    # Cria o diretório se não existir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ===== CARREGAR DADOS =====
    print("\n[1/4] Carregando dados...")
    # Lê o CSV com os descritores e a coluna Group
    df = pd.read_csv(descriptors_csv)
    
    print(f"  ✓ Descritores carregados: {len(df)} registros")
    
    # ===== FILTRAR DADOS VÁLIDOS =====
    print("\n[2/4] Filtrando dados válidos...")
    # Remover linhas sem ventrículo detectado (área = 0) ou sem grupo definido
    df_valid = df[(df['area'] > 0) & (df['Group'].notna())]
    
    print(f"  ✓ Registros válidos: {len(df_valid)}")
    print(f"  ✓ Classes encontradas: {sorted(df_valid['Group'].unique())}")
    
    # Mostrar quantos registros de cada classe
    for group in sorted(df_valid['Group'].unique()):
        count = (df_valid['Group'] == group).sum()
        print(f"     • {group}: {count} registros")
    
    # ===== DEFINIR CORES PARA CADA CLASSE =====
    # Mapa de cores conforme especificado no trabalho
    color_map = {
        'Converted': 'black',      # Preto para Converted
        'Nondemented': 'blue',     # Azul para Nondemented
        'Demented': 'red'          # Vermelho para Demented
    }
    
    # ===== LISTAR CARACTERÍSTICAS PARA PLOTAR =====
    # Os 6 descritores morfológicos calculados
    features = ['area', 'circularity', 'eccentricity', 'rectangularity', 'solidity', 'diameter']
    
    # ===== GERAR COMBINAÇÕES ÚNICAS DE PARES =====
    print("\n[3/4] Gerando combinações únicas de pares...")
    # combinations(features, 2) gera todas as combinações de 2 elementos
    # Isso resulta em C(6,2) = 15 combinações únicas
    # Exemplo: (area, circularity), (area, eccentricity), etc.
    # Não repete com ordem invertida: não teremos (circularity, area) se já temos (area, circularity)
    pairs = list(combinations(features, 2))
    
    print(f"  ✓ Total de gráficos a gerar: {len(pairs)}")
    print(f"  ✓ Combinações: C(6,2) = 15 pares únicos")
    
    # ===== GERAR SCATTERPLOTS =====
    print("\n[4/4] Gerando scatterplots...")
    
    # Para cada par de características
    for idx, (feat_x, feat_y) in enumerate(pairs, 1):
        print(f"  [{idx}/{len(pairs)}] {feat_x} vs {feat_y}")
        
        # Criar figura com tamanho adequado
        plt.figure(figsize=(10, 8))
        
        # Plotar cada classe com sua cor específica
        for group, color in color_map.items():
            # Filtrar dados apenas desta classe
            df_group = df_valid[df_valid['Group'] == group]
            
            # Se houver dados desta classe, plotar
            if len(df_group) > 0:
                plt.scatter(
                    df_group[feat_x],           # Valores do eixo X
                    df_group[feat_y],           # Valores do eixo Y
                    c=color,                    # Cor conforme o mapa
                    label=group,                # Label para a legenda
                    alpha=0.6,                  # Transparência (60% opaco)
                    s=50,                       # Tamanho dos pontos
                    edgecolors='black',         # Borda preta nos pontos
                    linewidth=0.5               # Espessura da borda
                )
        
        # ===== CONFIGURAR APARÊNCIA DO GRÁFICO =====
        # Definir label do eixo X com primeira letra maiúscula
        plt.xlabel(feat_x.capitalize(), fontsize=12, fontweight='bold')
        # Definir label do eixo Y com primeira letra maiúscula
        plt.ylabel(feat_y.capitalize(), fontsize=12, fontweight='bold')
        # Título do gráfico mostrando as duas características
        plt.title(f'{feat_x.capitalize()} vs {feat_y.capitalize()}', 
                  fontsize=14, fontweight='bold')
        # Adicionar legenda com as classes
        plt.legend(fontsize=10, loc='best')
        # Adicionar grade para facilitar leitura
        plt.grid(True, alpha=0.3)
        # Ajustar layout para não cortar labels
        plt.tight_layout()
        
        # ===== SALVAR FIGURA =====
        # Nome do arquivo: scatter_feature1_vs_feature2.png
        filename = f'scatter_{feat_x}_vs_{feat_y}.png'
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        # Fechar a figura para liberar memória
        plt.close()
    
    # ===== FINALIZAÇÃO =====
    print(f"\n{'='*80}")
    print("SCATTERPLOTS GERADOS COM SUCESSO!")
    print(f"{'='*80}")
    print(f"✓ Total de gráficos: {len(pairs)}")
    print(f"✓ Salvos em: {output_path}/")
    print(f"✓ Cores utilizadas:")
    print(f"   • Preto = Converted")
    print(f"   • Azul = Nondemented")
    print(f"   • Vermelho = Demented\n")


# ===============================================================================
# EXECUÇÃO PRINCIPAL
# ===============================================================================

if __name__ == "__main__":
    """
    Bloco principal de execução.
    
    Este bloco é executado quando o arquivo é rodado diretamente.
    Coordena todo o processo de:
    1. Processar o dataset completo (segmentar + calcular descritores)
    2. Gerar os scatterplots com as classes coloridas
    """
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "ANÁLISE COMPLETA DO DATASET OASIS" + " " * 26 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # ============================================================
    # CONFIGURAÇÕES
    # ============================================================
    
    # Diretório com as imagens sagitais (.nii.gz)
    # AJUSTE este caminho se necessário para apontar para suas imagens
    IMAGES_DIR = "sag/sag"
    
    # Arquivo CSV de saída com descritores e classes
    OUTPUT_CSV = "ventricle_descriptors_full.csv"
    
    # Arquivo CSV com dados demográficos fornecido (contém a coluna 'Group')
    DEMOGRAPHIC_CSV = "oasis_longitudinal_demographic.csv"
    
    # Diretório para salvar os 15 scatterplots
    SCATTERPLOT_DIR = "scatterplots"
    
    # Threshold para segmentação (ajuste entre 10-30 se necessário)
    THRESHOLD = 20
    
    # ============================================================
    # ETAPA 1: PROCESSAR DATASET COMPLETO
    # ============================================================
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "ETAPA 1: PROCESSAR DATASET" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Chama a função que processa todas as imagens do dataset
    # Retorna um DataFrame com todos os descritores e classes
    df_descriptors = process_full_dataset(
        images_dir=IMAGES_DIR,
        demographic_csv=DEMOGRAPHIC_CSV,
        output_csv=OUTPUT_CSV,
        threshold=THRESHOLD
    )
    
    # Verificar se o processamento foi bem-sucedido
    if df_descriptors is None:
        print("\n❌ Falha no processamento do dataset. Encerrando.")
        exit(1)
    
    # ============================================================
    # ETAPA 2: GERAR SCATTERPLOTS (ITEM 8)
    # ============================================================
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 23 + "ETAPA 2: GERAR SCATTERPLOTS" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Chama a função que gera os 15 scatterplots únicos
    # Usa o CSV gerado na etapa anterior
    generate_scatterplots(
        descriptors_csv=OUTPUT_CSV,
        output_dir=SCATTERPLOT_DIR
    )
    
    # ============================================================
    # FINALIZAÇÃO
    # ============================================================
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "ANÁLISE COMPLETA FINALIZADA!" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print(f"\n📊 RESULTADOS:")
    print(f"   • Descritores + Classes salvos em: {OUTPUT_CSV}")
    print(f"   • 15 Scatterplots salvos em: {SCATTERPLOT_DIR}/")
    print(f"   • Total de imagens processadas: {len(df_descriptors)}")
    print(f"\n✅ Trabalho concluído com sucesso!\n")