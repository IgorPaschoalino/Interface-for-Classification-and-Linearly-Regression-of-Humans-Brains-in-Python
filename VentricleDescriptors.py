# ------------------- Imports ---------------------

import cv2
import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd

# ------------------- Classe de Descritores ---------------------

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
    
    def calculate_all_descriptors(self, binary_mask):
        """
        Calcula todos os 6 descritores para uma máscara binária dos ventrículos.
        Esta é a função principal que chama todas as outras funções de cálculo.
        
        Recebe:
        - binary_mask: Imagem binária (True/False ou 0/255) com os ventrículos segmentados
        
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
            print("AVISO: Nenhum contorno encontrado na máscara!")
            return self._empty_descriptors()
        
        # Usar o maior contorno (assumindo que é o ventrículo principal)
        # Isso garante que pegamos o ventrículo e não algum ruído pequeno
        main_contour = max(contours, key=cv2.contourArea)
        
        print("\n[Descritores] Contorno principal encontrado com {} pontos".format(len(main_contour)))
        
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
        area = cv2.contourArea(contour)
        print(f"  [1/6] Área calculada: {area:.2f} pixels")
        return area
    
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
            print("  [2/6] AVISO: Perímetro zero detectado!")
            return 0
        
        # Aplica a fórmula: 4 * pi * área / perímetro²
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        print(f"  [2/6] Circularidade calculada: {circularity:.4f}")
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
                print("  [3/6] AVISO: Contorno muito pequeno, usando excentricidade = 1")
                return 1
            # Retorna a razão entre largura e altura
            eccentricity = max(w, h) / min(w, h)
            print(f"  [3/6] Excentricidade calculada (bounding box): {eccentricity:.4f}")
            return eccentricity
        
        try:
            # Ajusta uma elipse no contorno
            # ellipse = ((centro_x, centro_y), (eixo_maior, eixo_menor), ângulo)
            ellipse = cv2.fitEllipse(contour)
            # Pega o maior e o menor eixo da elipse
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            
            # Evita divisão por zero
            if minor_axis == 0:
                print("  [3/6] AVISO: Eixo menor = 0, usando excentricidade = 1")
                return 1
            
            # Calcula a excentricidade como a razão entre os eixos
            eccentricity = major_axis / minor_axis
            print(f"  [3/6] Excentricidade calculada (elipse): {eccentricity:.4f}")
            return eccentricity
        except:
            # Se houver erro ao ajustar a elipse, retorna 1 (forma circular)
            print("  [3/6] ERRO ao ajustar elipse, usando excentricidade = 1")
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
            print("  [4/6] AVISO: Área do retângulo = 0!")
            return 0
        
        # Calcula a retangularidade: área do ventrículo / área do retângulo
        rectangularity = area / rect_area
        print(f"  [4/6] Retangularidade calculada: {rectangularity:.4f}")
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
            print("  [5/6] AVISO: Área do fecho convexo = 0!")
            return 0
        
        # Calcula a solidez: área do ventrículo / área do fecho convexo
        solidity = area / hull_area
        print(f"  [5/6] Solidez calculada: {solidity:.4f}")
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
            print("  [6/6] AVISO: Menos de 2 pontos no contorno!")
            return 0
        
        # Se o contorno tiver muitos pontos (>100), faz amostragem para otimizar
        # Calcular distância entre todos os pares de 1000 pontos seria muito lento
        if len(points) > 100:
            # Seleciona 100 pontos igualmente espaçados ao longo do contorno
            indices = np.linspace(0, len(points)-1, 100, dtype=int)
            points = points[indices]
            print(f"  [6/6] Amostrando {len(points)} pontos do contorno para otimização")
        
        # Calcula a matriz de distâncias entre todos os pares de pontos
        # pdist retorna um vetor com todas as distâncias par a par
        distances = pdist(points, metric='euclidean')
        # Pega a maior distância encontrada (esse é o diâmetro)
        diameter = np.max(distances)
        print(f"  [6/6] Diâmetro calculado: {diameter:.2f} pixels")
        
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
    
    def print_descriptors(self, descriptors):
        """
        Imprime os descritores de forma organizada e legível.
        Separa os obrigatórios dos adicionais e mostra informações sobre cada um.
        
        Recebe:
        - descriptors: dicionário com os descritores calculados
        """
        print("\n" + "=" * 60)
        print("DESCRITORES MORFOLÓGICOS DOS VENTRÍCULOS")
        print("=" * 60)
        
        print("\n>>> DESCRITORES OBRIGATÓRIOS:")
        print(f"  1. Área:           {descriptors['area']:.2f} pixels")
        print(f"     └─ Slide 13: Quantidade de pixels do ventrículo")
        print(f"  2. Circularidade:  {descriptors['circularity']:.4f}")
        print(f"     └─ Slide 13: 4πA/P² (1.0 = círculo, <1 = irregular)")
        print(f"  3. Excentricidade: {descriptors['eccentricity']:.4f}")
        print(f"     └─ Slide 14: Eixo maior/menor (~1 = circular, >1 = alongado)")
        
        print("\n>>> DESCRITORES ADICIONAIS:")
        print(f"  4. Retangularidade: {descriptors['rectangularity']:.4f}")
        print(f"     └─ Slide 14: Área/Área Retângulo (1.0 = preenche, <1 = irregular)")
        print(f"  5. Solidez:         {descriptors['solidity']:.4f}")
        print(f"     └─ Slide 19: Área/Fecho Convexo (1.0 = convexo, <1 = côncavo)")
        print(f"  6. Diâmetro:        {descriptors['diameter']:.2f} pixels")
        print(f"     └─ Slide 13: Maior distância entre pontos do contorno")
        
        print("\n" + "=" * 60)
    
    def save_to_csv(self, descriptors, output_path, image_name=None):
        """
        Salva os descritores em arquivo CSV.
        Útil para análise posterior ou integração com outros dados clínicos.
        
        Recebe:
        - descriptors: dicionário com os descritores calculados
        - output_path: caminho do arquivo CSV a ser criado
        - image_name: nome da imagem (opcional)
        """
        # Adiciona informação sobre a imagem se fornecida
        data = {'image_name': image_name} if image_name else {}
        data.update(descriptors)
        
        # Cria DataFrame e salva
        df = pd.DataFrame([data])
        df.to_csv(output_path, index=False)
        print(f"\n✓ Descritores salvos em: {output_path}")


# Exemplo de uso
if __name__ == "__main__":
    # Este arquivo normalmente não é executado diretamente
    # Ele é importado pelo main.py que coordena segmentação + descrição
    
    print("=" * 60)
    print("MÓDULO DE EXTRAÇÃO DE DESCRITORES")
    print("=" * 60)
    print("\nEste módulo calcula 6 descritores morfológicos:")
    print("  Obrigatórios: Área, Circularidade, Excentricidade")
    print("  Adicionais: Retangularidade, Solidez, Diâmetro")
    print("\nPara usar este módulo, importe-o no seu código principal:")
    print("  from descriptors import VentricleDescriptors")
    print("\nOu execute o arquivo main.py para análise completa.")
    print("=" * 60)