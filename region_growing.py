# ------------------- Imports ---------------------

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------- Algoritmo ---------------------

class VentricleSegmentation:

    # Construtor do objeto que vai fazer as segmentações
    # Tem como váriaveis:
    # image_path -> caminho da imagem (talvez teremos de mudar)
    # img_data -> dados da imagem (não entendi se é a propria imagem ou valores binario ou algo do tipo, mas eu sei que essa imagem não é visualizavel facilmente não, bglh é diferente mesmo)
    # slice_idx ->
    # seed ->
    # brain_mask ->
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
    # Carrega a imagem e guarda ela em nii_img e depois em img_data guarda as informações da imagem (quaisquer que elas sejam)
    # Printa o img_data.shape, que nada mais é do que o tamanho dela altura x largura
    # Verifica se a imagem é 2D ou 3D e se for 3D transforma ela em imagem sagital (no caso todas as nossas imagens são sagitais, então essa parte de transformar de 3D pra 2D não serve pra nada)
    def load_image(self):
        """Carrega a imagem NIfTI e seleciona o corte sagital central."""
        nii_img = nib.load(self.image_path)
        self.img_data = nii_img.get_fdata()
        
        print(f"Shape da imagem carregada: {self.img_data.shape}")
        
        # Verificar dimensionalidade da imagem
        if self.img_data.ndim == 2:
            # Imagem já é 2D
            print("Imagem 2D detectada")
            return self.img_data
        elif self.img_data.ndim == 3:
            # Imagem 3D - seleciona corte sagital central
            print("Imagem 3D detectada - selecionando corte central")
            self.slice_idx = self.img_data.shape[0] // 2
            return self.img_data[self.slice_idx, :, :]
        else:
            raise ValueError(f"Dimensionalidade não suportada: {self.img_data.ndim}D")
    



    # Função que normaliza a imagem (oq quer que isso signifique)
    def normalize_image(self, img):
        """Normaliza a imagem para range [0, 255]."""
        img_norm = img - np.min(img)
        img_norm = img_norm / (np.max(img_norm) + 1e-8) * 255
        return img_norm.astype(np.uint8)
    



    # Como o fundo tem a msm cor do Ventrículo lateral, se faz necessário analisar só o cerebro, pq se não fica foda de selecionar a seed corretamente, essa função tem esse papel de separar o cerebro do fundo pra dps a gente escolher a seed
    # Faz o processo por segmentação por limiarização utilizando ostu pra encontrar a melhor separação entre o objeto (cerebro) e o fundo que queremos remover
    def create_brain_mask(self, img):
        """
        Cria uma máscara do cérebro para excluir o fundo.
        
        Args:
            img: Imagem 2D normalizada
            
        Returns:
            Máscara binária do cérebro
        """

        # Usar threshold de Otsu para separar cérebro do fundo
        # Faz Otsu bem automático msm
        threshold = filters.threshold_otsu(img)
        

        # Os ventriculos tem tons de cinza escuros como o fundo, porém o fundo é mais escuro, o ponto de usar apenas metade do threshold  é excluir apenas os pontos mais escuros da imagem, pois se usassemos o threshold sem essa multiplicação por 0,5 existe uma chance do vantriculo ser considerado como fundo ao inves de parte do objeto, o que não é o que queremos.
        brain_mask = img > threshold * 0.5
        
        # Morfologia matématica pra tirar os ruídos e corrigir uns BO da imagem pós limiarização. Faz nesse caso a abertura, depois o fechamento e depois tampa buraco xD (necessário ver depois o por que de fazer nessa ordem de abertura e fechamento e por que usar elemento estruturante de 5,5 e 11,11)
        brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((5, 5)))
        brain_mask = ndimage.binary_closing(brain_mask, structure=np.ones((11, 11)))
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        
        # Remover pequenas componentes (ruído). Faz todo um processo numerando pequenos objetos que ficaram separados do objeto principal (cerbro) e marca todos os objetos da imagem com lables (inclusive o cerebro). Passa por todos os lables verificando e guardando em um vetor o tamanho deles em pixels e depois pega o lable que tiver mais pixels (objeto principal vulgo cerebro). Depois coloca na mascara (brain_mask) apenas os componentes que tem lable iguais ao lable de maior quantidade de pixels 
        labeled, num_features = ndimage.label(brain_mask)
        if num_features > 0:
            sizes = ndimage.sum(brain_mask, labeled, range(1, num_features + 1))
            # Manter apenas a maior componente (o cérebro)
            max_label = np.argmax(sizes) + 1
            brain_mask = labeled == max_label
        
        # Faz uma erosão pra garantir que ta pegando dentro do cerbro msm
        brain_mask = ndimage.binary_erosion(brain_mask, structure=np.ones((15, 15)))
        
        # Retorna a mascara que define o que é cerebro na imagem
        return brain_mask
    



    # Função que vai gerar várias seeds e vai no final escolher qual é a melhor seed para definir o ventrículo dentre as seeds criadas
    # cria as seeds dentro da região do cerebro (já excluímos o fundo) colocando as seeds em regiões mais escuras (tons de cinza mais escuros). Nesse caso ta gerando 50 seeds pra escolher uma só
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
        # Coloca a altura e largura  da imagem nas variaveis h e w
        h, w = img.shape
        # Cria um array vazio de candidatos a seed
        candidates = []
        
        # Cria uma cópia da imagem e nessa cópia faz tudo que não faz parte do cerebro (definido na brain_mask) ganha o valor 255 (referente ao branco). como um dos parâmetros para definir a seed é o quão escura é a região, o fundo não será mais escolhido, uma vez que é completamente branco
        img_masked = img.copy() # faz uma cópia da imagem pra gerar a mascara
        img_masked[~brain_mask] = 255  # Fundo vira branco (alta intensidade)
        
        # Aplicar threshold para encontrar regiões escuras DENTRO do cérebro
        # Ventrículos são as regiões MAIS ESCURAS dentro do cérebro
        dark_threshold = np.percentile(img_masked[brain_mask], 15)  # retorna um threshold no qual 15% dos meus dados estão abaixo (no caso são mais escuros)
        dark_regions = (img_masked < dark_threshold) & brain_mask # define as regiões escuras do cerebro pegando apenas regiões menores que o theshold (dark_threshold) e que fazem parte da brain_mask
        
        # Aplicar morfologia para limpar ruído (no caso faz abertura depois fechamento). Se faz necessário entender por que ele está fazendo essas operações nessa ordem e com os elementos estruturantes 3,3 e 5,5
        dark_regions = ndimage.binary_opening(dark_regions, structure=np.ones((3,3)))
        dark_regions = ndimage.binary_closing(dark_regions, structure=np.ones((5,5)))
        
        # Calcular distância das bordas das regiões escuras (o que ele faz é retornar a distância euclidiana de todos os pontos classificados como objeto em relação ao fundo)
        distance_map = ndimage.distance_transform_edt(dark_regions)
        
        # Encontrar máximos locais na distância (centro das regiões escuras) e define os candidatos a seed verificando se a distância deles até o fundo for mais que 3 pixels
        local_max = filters.rank.maximum(distance_map.astype(np.uint8), np.ones((15, 15)))
        seed_candidates = (distance_map == local_max) & (distance_map > 3)
        
        # Obter coordenadas dos candidatos
        y_coords, x_coords = np.where(seed_candidates)
        
        # Printa quantos candidatos encontrou
        print(f"DEBUG: Encontrados {len(y_coords)} candidatos potenciais")
        
        for y, x in zip(y_coords, x_coords):
            # VERIFICAÇÃO CRÍTICA: Garantir que está dentro da máscara do cérebro
            if not brain_mask[y, x]:
                continue
            
            # Definir região de análise ao redor do candidato (delimita onde começa e onde termnina a janela de análise garantindo que vai estar dentro da imagem, tamaho da janela é 15 pra todos os lados em relação ao ponto candidato)
            window_size = 15
            y_min = max(0, y - window_size)
            y_max = min(h, y + window_size)
            x_min = max(0, x - window_size)
            x_max = min(w, x + window_size)
            
            # Pega a janela que vai ser análisada tanto na imagem quanto na mascara do cerebro (pra fazer a analise em ambos e garantir que no final a parte da janela que estamos analisando esta dentro do cerebro de fato)
            window = img[y_min:y_max, x_min:x_max]
            window_mask = brain_mask[y_min:y_max, x_min:x_max]
            
            # Calcular características apenas na região do cérebro (eu acho que isso aq tirar nao ia mudar nada tlg)
            if np.sum(window_mask) < 50:  # Região muito pequena
                continue
            
            # Coloca na analise da janela apenas as partes que estão dentro do cerebro
            window_values = window[window_mask]
            
            # Calcular características
            features = {
                'position': (y, x), # posição
                'intensity': float(img[y, x]), # o quao claro é a seed
                'mean_intensity': float(np.mean(window_values)), # intensidade media da região
                'std_intensity': float(np.std(window_values)), # desvio padrão da janela 
                'homogeneity': float(1.0 / (1.0 + np.std(window_values))), # define se é homogeneo ou heterogeneo usando o DP
                'centrality': 1.0 - np.sqrt(((y - h/2)/h)**2 + ((x - w/2)/w)**2), # Posição relativa (ventrículos geralmente na região central do cérebro)
                'distance_from_edge': float(distance_map[y, x]), # distância em relação ao fim da região escura
                'distance_from_brain_edge': float(ndimage.distance_transform_edt(brain_mask)[y, x]) # Distância da borda do cérebro (ventrículos estão no interior)
            }
            
            # Coloca num array as caracteristicas de cada seed que foi encontrada, essas caracteristicas vão ser analisadas em outra função pra definir a melhor seed
            candidates.append(features)
        
        # Retorna a lista de seeds
        return candidates[:n_candidates]
    


    # Função que vai ponderar e dar notas para cada uma das seeds, a que tiver melhor nota depois provavelmente vai ser escolhida como seed do ventriculo. Mas essa função só da as notas, quem escolhe é outra
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
        # se a intensidade é alta (muito perto do branco) intensity_score fica mais proximo de 0, se intensidade muito baixa (muito perto do preto) intensity_score fica mais proximo de 1. Soma o intensity_score ao score total (com peso de 3,5)
        intensity_score = 1.0 - (features['intensity'] / 255.0) 
        score += intensity_score * 3.5
        
        # Homogeneidade (região uniforme)
        # Verifica se a região é coerente com a cor da seed, isto é, não tem muita variação de cores nas seeds da região (DP baixo se não me engano). Soma o a homogeniedade ao score total (com peso de 2,5) (verificar depois se os valores de homogeniedade estão sempre entre 0 e 1 ou não, acho que não mas precisa conferir na linha 190)
        score += features['homogeneity'] * 2.5
        
        # Centralidade (ventrículos próximos ao centro)
        # Verifica se a região está proxima do centro do cerebro (o resultado esta entre 0 e 1 e o calculo é feito na linha 191). Soma a centralidade ao score total (com peso de 2)
        score += features['centrality'] * 2.0
        
        # Distância da borda da região escura
        # Verifica o quão longe a seed está do fim da região escura a qual ela pertence. Se a distância é maior que 20 pixels até a borda da região escura, o score será de 1.5, se for menor do que 20 pixels, será esse valor/20 * 1.5 (que é menor sempre do que 1.5)
        score += min(features['distance_from_edge'] / 20.0, 1.0) * 1.5
        
        # Distância da borda do cérebro. Se a distancia for maior que 30, recebe score 4. Se tiver distância menor que 30 pixels, recebe distancia/30 * 4 (obrigatoriamente menor que, logo o cria foi penalisado)
        score += min(features['distance_from_brain_edge'] / 30.0, 1.0) * 4.0
        
        # Retorna o Score total de cada seed candidata
        return score
    

    # Selecionar a melhor seed com base nos scores calculados para cada seed candidata e retorna o que foi considerada a melhor seed (ventriculo lateral)
    # Essa função é quem chama a função de gerar o score para cada seed
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
        
        # Calcular scores. criar um array de scores e seeds (Score, posição da seed) com os resultados de cada candidato 
        scored_candidates = [(self.score_seed(c), c['position']) for c in candidates]
        # Ordena os scores em ordem decrescente (como é uma tupla tem que selecionar qual parte da tupla escolher pra ordenar, no caso a de indice 0, vulgo o score)
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Debug: mostrar top 5 candidatos
        print("\nDEBUG: Top 5 candidatos:")
        for i, (score, pos) in enumerate(scored_candidates[:5]):
            print(f"  {i+1}. Score: {score:.2f}, Posição: {pos}")
        
        # Pega o elemento com index 0 da lista (O de melhor score) e retorna o segundo valor da tupla [1], no caso a posição da seed
        return scored_candidates[0][1]
    

    # Essa função detem o algoritmo de crescimento de regiões e recebe a imagem, a seed e o threshold
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

        # Guarda nas variaveis h e w a altura e largura da imagem
        h, w = img.shape
        # Cria uma matriz para guardar o resultado com a mesma altura e mesma largura da imagem original sendo de tipo booleano (já que vai ser só objeto e fundo)
        segmented = np.zeros((h, w), dtype=bool)
        
        # Se não tiver seed, retorna a matriz vazia (não tem ventriculo pra chamar de objeto)
        if seed is None:
            return segmented
        
        # Pega o valor do pixel da seed (o tom de cinza dela entre 0 e 255)
        seed_value = img[seed[0], seed[1]]
        
        # Cria uma fila para processar pixels
        queue = [seed]
        # Marca a seed como parte do ventriculo
        segmented[seed[0], seed[1]] = True
        
        # 8-conectividade (pixels ao redor, vizinhos ou derivados disso)
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Enquanto ainda ouverem pixels a serem analisados
        while queue:
            # Remove o pixel da fila e guarda suas coordenadas em x e y
            y, x = queue.pop(0)
            
            # Para cada vizinho (-1,-1) (-1,0) (-1,1) etc...
            for dy, dx in neighbors:
                # Pega a posição real do vizinho (coordenada do pixel + vizinho((-1,-1) (-1,0) (-1,1)))
                ny, nx = y + dy, x + dx
                
                # Verificar limites (se ta dentro da imagem e não faz parte do ventriculo ainda...)
                if 0 <= ny < h and 0 <= nx < w and not segmented[ny, nx]:
                    # CRÍTICO: Verificar se está dentro da máscara do cérebro
                    # Se a mascara do cerebro existir mas o pixel em questão não está dentro do cerebro...
                    if self.brain_mask is not None and not self.brain_mask[ny, nx]:
                        continue
                    
                    # Se estiver dentro do cerebro...
                    # Verificar similaridade
                    # Se o pixel analisado tiver diferença em tons de cinza menor que o threshold (ainda é considerado objeto)
                    if abs(float(img[ny, nx]) - float(seed_value)) < threshold:
                        # O pixel vira Ventriculo
                        segmented[ny, nx] = True
                        # O pixel entra na Fila para encontrar seus vizinhos
                        queue.append((ny, nx))
        
        # Pós-processamento: remover pequenas regiões e preencher buracos
        # Depois de fazer tudo isso e definir o ventriculo, pega o resultado e faz uma abertura seguida de fechamento e tapar buracos (se faz necessário descobrir e entender o por que fazer abertura e depois fechamento com os objetos estruturantes 3,3 e 5,5)
        segmented = ndimage.binary_opening(segmented, structure=np.ones((3,3)))
        segmented = ndimage.binary_closing(segmented, structure=np.ones((5,5)))
        segmented = ndimage.binary_fill_holes(segmented)
        
        return segmented
    
    # Guarda os resultados em formato de imagem png se nao me engano (tem que ver de guardar no formato complexo la tbm que aparentemente ele é mto importante pra essas paradas. Essa função com certeza vai ser alterada para se encaixar na interface depois)
    def save_results(self, img, segmentation, seed, output_dir):
        """
        Salva 4 imagens: original, máscara do cérebro, overlay e máscara da segmentação.
        
        Args:
            img: Imagem original
            segmentation: Máscara de segmentação
            seed: Posição da seed selecionada
            output_dir: Diretório de saída
        """
        output_dir = Path(output_dir) # Caminho para gguardar os resultados
        output_dir.mkdir(parents=True, exist_ok=True) #  Cria esse diretorio se ele nao existir
        
        base_name = self.image_path.stem.replace('.nii', '') # Serve pra normalizar o nome (tirar o .nii), não sei se isso vai poder ficar aqui no final da matematica, vamos descobrir
        
        # Normalizar imagem para visualização (passa a imagem na função de normalização que vai deixar ela em tons de 0 a 255)
        img_vis = self.normalize_image(img)
        
        # 1. Imagem original 
        plt.figure(figsize=(10, 10))
        plt.imshow(img_vis, cmap='gray')
        plt.axis('off')
        plt.title('Imagem Original', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{base_name}_original.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # 2. Máscara do cérebro (DEBUG)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_vis, cmap='gray')
        if self.brain_mask is not None:
            plt.imshow(self.brain_mask, cmap='Greens', alpha=0.3)
        plt.axis('off')
        plt.title('Máscara do Cérebro', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{base_name}_brain_mask.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # 3. Apenas segmentação (máscara)
        plt.figure(figsize=(10, 10))
        plt.imshow(segmentation, cmap='Reds', alpha=1.0)
        plt.axis('off')
        plt.title('Máscara da Segmentação', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{base_name}_mask.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # 4. Original + Overlay + Seed
        plt.figure(figsize=(10, 10))
        plt.imshow(img_vis, cmap='gray')
        plt.imshow(segmentation, cmap='Reds', alpha=0.5)
        if seed is not None:
            plt.plot(seed[1], seed[0], 'b*', markersize=20, markeredgewidth=2, 
                    markeredgecolor='yellow', label='Seed')
            plt.legend(fontsize=12)
        plt.axis('off')
        plt.title('Segmentação Final', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{base_name}_overlay.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"\nImagens salvas em: {output_dir}")
        print(f"  - {base_name}_original.png")
        print(f"  - {base_name}_brain_mask.png (DEBUG)")
        print(f"  - {base_name}_mask.png")
        print(f"  - {base_name}_overlay.png")
    

    # Função que é chamada para fazer todo o processo de uma vez só
    def process(self, output_dir='results', threshold=20):
        """
        Executa todo o pipeline de segmentação.
        
        Args:
            output_dir: Diretório para salvar resultados
            threshold: Limiar para crescimento de regiões
        """
        print("=" * 60)
        print("INICIANDO SEGMENTAÇÃO DE VENTRÍCULOS LATERAIS")
        print("=" * 60)
        
        print("\n[1/6] Carregando imagem...")
        # Carrega a imagem por meio de nossa função load_image
        img = self.load_image()
        # Carrega a imagem por meio de nossa função load_image
        img_norm = self.normalize_image(img)
        
        print("\n[2/6] Criando máscara do cérebro...")
        # Cria a mascara do cerebro chamando nossa função passando como parâmetro a imagem ja normalizada
        self.brain_mask = self.create_brain_mask(img_norm)
        # Soma a quantidade de pixels do cerebro
        brain_pixels = np.sum(self.brain_mask)
        print(f"  Pixels do cérebro: {brain_pixels}")
        
        print("\n[3/6] Gerando candidatos a seed...")
        # Cria a lista de candidatos a seed, chamando nossa função, passando como parâmetro a imagem ja normalizada e a mascara do cerebro
        candidates = self.generate_candidate_seeds(img_norm, self.brain_mask)
        print(f"  Candidatos válidos: {len(candidates)}")
        
        print("\n[4/6] Selecionando melhor seed...")
        # Seleciona a melhor seed, que será usada para realizar o crescimento de regiões, por meio da nossa função de seleção de seed
        self.seed = self.select_best_seed(candidates)
        
        # Se não tiver encontrado nenhuma seed usa o centro do cerebro como seed
        if self.seed is None:
            print("  AVISO: Nenhuma seed encontrada!")
            print("  Usando posição central dentro do cérebro...")
            # Encontrar centro de massa do cérebro
            if self.brain_mask is not None:
                y_coords, x_coords = np.where(self.brain_mask)
                if len(y_coords) > 0:
                    self.seed = (int(np.mean(y_coords)), int(np.mean(x_coords)))
                else:
                    h, w = img.shape
                    self.seed = (h // 2, w // 2)
        
        print(f"  Seed selecionada: {self.seed}")
        print(f"  Intensidade na seed: {img_norm[self.seed[0], self.seed[1]]}")
        
        print("\n[5/6] Executando crescimento de regiões...")
        # Executa o crescimento de regiões passando como parâmetro a imagem normalizada, a seed encontrada, e um threshold pre definido como 20
        segmentation = self.region_growing(img_norm, self.seed, threshold)
        # Número de pixels no ventriculo
        seg_pixels = np.sum(segmentation)
        print(f"  Pixels segmentados: {seg_pixels}")
        
        print("\n[6/6] Salvando resultados...")
        # Salva os resultados em imagens por meio da nossa função de salvar resultados
        self.save_results(img, segmentation, self.seed, output_dir)
        
        print("\n" + "=" * 60)
        print("SEGMENTAÇÃO CONCLUÍDA!")
        print("=" * 60)
        
        # Retorna a segmentação
        return segmentation



# Exemplo de uso
if __name__ == "__main__":
    # Caminho para sua imagem
    image_path = "sag/sag/OAS2_0001_MR1_sag.nii.gz"
    
    # Criar segmentador (objeto com base na classe VentricleSegmentation) passando como parâmetro o caminho da imagem a ser segmentada
    # Talvez adaptar para receber a imagem em questão ao invés do caminho (temos que ver)
    segmenter = VentricleSegmentation(image_path)
    
    # Processar (função da classe VentricleSegmentation que faz a segmentação de fato, recebe o diretório pra guardar as imagem de resultado e o limite de crescimento da região (adaptar se estiver ficando muito bosta))
    # Talvez adaptar para ao invés de guardar os resultados, enviar pra outro código python ou até mesmo printar na tela, tem que ver como vai ser a interface
    segmentation = segmenter.process(
        output_dir='results',
        threshold=20  # Valor que define o limite de crescimento da região, ajuste este valor se necessário (10-30 funciona bem)
    )
    
    print(f"\nÁrea segmentada: {np.sum(segmentation)} pixels")
    print(f"Porcentagem do cérebro: {100 * np.sum(segmentation) / np.sum(segmenter.brain_mask):.2f}%")