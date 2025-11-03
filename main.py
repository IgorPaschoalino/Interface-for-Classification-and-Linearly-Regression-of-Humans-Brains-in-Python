# ------------------- Imports ---------------------

# Importa as classes dos outros arquivos
from VentricleSegmentation import VentricleSegmentation
from VentricleDescriptors import VentricleDescriptors

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------- Função Principal ---------------------

def analyze_ventricle(image_path, threshold=20, save_images=True, output_dir='results'):
    """
    Função principal que coordena TODO O PROCESSO:
    1. Segmentação dos ventrículos laterais (usando segmentation.py)
    2. Extração dos 6 descritores morfológicos (usando descriptors.py)
    3. Salvamento dos resultados
    
    Recebe:
    - image_path: caminho para o arquivo de imagem NIfTI (.nii.gz)
    - threshold: limite de diferença de intensidade para crescimento de regiões (padrão: 20)
    - save_images: se True, salva imagens de visualização (padrão: True)
    - output_dir: diretório onde salvar os resultados (padrão: 'results')
    
    Retorna:
    - Tupla (segmentation, descriptors, img_normalized)
      * segmentation: máscara binária dos ventrículos
      * descriptors: dicionário com os 6 descritores
      * img_normalized: imagem original normalizada
    """
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 10 + "ANÁLISE COMPLETA DE VENTRÍCULOS" + " " * 17 + "█")
    print("█" + " " * 15 + "Segmentação + Descritores" + " " * 20 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")
    
    # ============================================================
    # ETAPA 1: SEGMENTAÇÃO DOS VENTRÍCULOS
    # ============================================================
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 18 + "ETAPA 1: SEGMENTAÇÃO" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Criar o objeto segmentador passando o caminho da imagem
    segmenter = VentricleSegmentation(image_path)
    
    # Executar a segmentação (chama o process() do arquivo segmentation.py)
    # Isso vai retornar a máscara binária dos ventrículos
    segmentation = segmenter.process(
        output_dir=output_dir,
        threshold=threshold
    )
    
    # Pegar a imagem normalizada que foi processada
    # (necessária para salvar resultados depois)
    img_normalized = segmenter.normalize_image(segmenter.load_image())
    
    # Verificar se a segmentação encontrou algum ventrículo
    seg_pixels = np.sum(segmentation)
    if seg_pixels == 0:
        print("\n⚠ AVISO: Nenhum ventrículo foi segmentado!")
        print("Tente ajustar o parâmetro 'threshold' (valores entre 10-30)")
        return None, None, None
    
    # ============================================================
    # ETAPA 2: EXTRAÇÃO DE DESCRITORES
    # ============================================================
    print("\n╔" + "═" * 58 + "╗")
    print("║" + " " * 13 + "ETAPA 2: EXTRAÇÃO DE DESCRITORES" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Criar o objeto extrator de descritores
    descriptor_extractor = VentricleDescriptors()
    
    print("[1/2] Calculando descritores morfológicos...")
    # Calcular todos os 6 descritores usando a máscara de segmentação
    # Isso vai chamar calculate_all_descriptors() do arquivo descriptors.py
    descriptors = descriptor_extractor.calculate_all_descriptors(segmentation)
    print("  ✓ Descritores calculados com sucesso!")
    
    print("\n[2/2] Exibindo resultados...")
    # Imprime os descritores de forma organizada
    descriptor_extractor.print_descriptors(descriptors)
    
    # ============================================================
    # ETAPA 3: SALVAR IMAGENS COM DESCRITORES (OPCIONAL)
    # ============================================================
    if save_images:
        print("\n╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "ETAPA 3: SALVANDO IMAGENS" + " " * 17 + "║")
        print("╚" + "═" * 58 + "╝\n")
        
        # Salvar imagem overlay com os descritores anotados
        save_overlay_with_descriptors(
            segmenter, 
            img_normalized, 
            segmentation, 
            descriptors, 
            output_dir
        )
    
    # ============================================================
    # ETAPA 4: SALVAR DESCRITORES EM CSV
    # ============================================================
    output_csv = Path(output_dir) / 'descriptors.csv'
    image_name = Path(image_path).stem
    descriptor_extractor.save_to_csv(descriptors, output_csv, image_name)
    
    # ============================================================
    # FINALIZAÇÃO
    # ============================================================
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 15 + "ANÁLISE CONCLUÍDA COM SUCESSO!" + " " * 12 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")
    
    print(f"✓ Área do ventrículo: {descriptors['area']:.0f} pixels")
    print(f"✓ Imagens salvas em: {output_dir}/")
    print(f"✓ Descritores salvos em: {output_csv}")
    
    # Retorna os resultados para uso posterior se necessário
    return segmentation, descriptors, img_normalized


def save_overlay_with_descriptors(segmenter, img_normalized, segmentation, descriptors, output_dir):
    """
    Salva uma imagem overlay especial com os descritores anotados.
    Essa função complementa as imagens já salvas pela segmentação.
    
    Recebe:
    - segmenter: objeto VentricleSegmentation (para pegar info da seed)
    - img_normalized: imagem normalizada
    - segmentation: máscara de segmentação
    - descriptors: dicionário com os descritores calculados
    - output_dir: diretório onde salvar
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extrai o nome base do arquivo
    base_name = segmenter.image_path.stem.replace('.nii', '')
    
    # Criar figura com a imagem overlay + descritores
    plt.figure(figsize=(12, 10))
    plt.imshow(img_normalized, cmap='gray')
    plt.imshow(segmentation, cmap='Reds', alpha=0.5)
    
    # Adicionar a seed se existir
    if segmenter.seed is not None:
        plt.plot(segmenter.seed[1], segmenter.seed[0], 'b*', 
                markersize=20, markeredgewidth=2, markeredgecolor='yellow', label='Seed')
        plt.legend(fontsize=12, loc='upper right')
    
    plt.axis('off')
    plt.title('Segmentação Final com Descritores Morfológicos', fontsize=16, pad=20)
    
    # Adicionar os descritores como texto na imagem
    # Criar um box de texto formatado com todos os descritores
    desc_text = (
        "═══ DESCRITORES OBRIGATÓRIOS ═══\n"
        f"Área:           {descriptors['area']:.2f} px\n"
        f"Circularidade:  {descriptors['circularity']:.4f}\n"
        f"Excentricidade: {descriptors['eccentricity']:.4f}\n"
        "\n"
        "═══ DESCRITORES ADICIONAIS ═══\n"
        f"Retangularidade: {descriptors['rectangularity']:.4f}\n"
        f"Solidez:         {descriptors['solidity']:.4f}\n"
        f"Diâmetro:        {descriptors['diameter']:.2f} px"
    )
    
    # Posicionar o texto no canto superior esquerdo
    plt.text(0.02, 0.98, desc_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=1),
            family='monospace')
    
    plt.tight_layout()
    output_file = output_dir / f'{base_name}_descriptors_overlay.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ {base_name}_descriptors_overlay.png")


# ------------------- Execução Principal ---------------------

if __name__ == "__main__":
    """
    Bloco principal de execução.
    Aqui você define o caminho da sua imagem e executa a análise completa.
    """
    
    # ============================================================
    # CONFIGURAÇÕES
    # ============================================================
    
    # Caminho para sua imagem NIfTI
    # ALTERE ESTE CAMINHO para apontar para sua imagem
    image_path = "sag/sag/OAS2_0001_MR1_sag.nii.gz"
    
    # Parâmetros de segmentação
    threshold = 20  # Ajuste entre 10-30 se necessário
    
    # Diretório de saída
    output_dir = 'results'
    
    # ============================================================
    # EXECUTAR ANÁLISE COMPLETA
    # ============================================================
    
    # Chama a função principal que faz tudo:
    # 1. Segmentação (usando segmentation.py)
    # 2. Extração de descritores (usando descriptors.py)
    # 3. Salvamento dos resultados
    segmentation, descriptors, img_normalized = analyze_ventricle(
        image_path=image_path,
        threshold=threshold,
        save_images=True,
        output_dir=output_dir
    )
    
    # ============================================================
    # RESULTADOS DISPONÍVEIS PARA USO POSTERIOR
    # ============================================================
    
    if segmentation is not None:
        print("\n" + "=" * 60)
        print("RESUMO DOS RESULTADOS")
        print("=" * 60)
        print(f"\n📊 Estatísticas da Segmentação:")
        print(f"   • Pixels segmentados: {np.sum(segmentation)}")
        print(f"   • Threshold usado: {threshold}")
        
        print(f"\n📏 Descritores Principais:")
        print(f"   • Área: {descriptors['area']:.2f} pixels")
        print(f"   • Circularidade: {descriptors['circularity']:.4f}")
        print(f"   • Diâmetro: {descriptors['diameter']:.2f} pixels")
        
        print(f"\n💾 Arquivos Gerados:")
        print(f"   • Imagens: {output_dir}/*.png")
        print(f"   • CSV: {output_dir}/descriptors.csv")
        
        print("\n" + "=" * 60)
    else:
        print("\n❌ Análise falhou. Verifique os parâmetros e tente novamente.")
    
    # ============================================================
    # EXEMPLO: PROCESSAR MÚLTIPLAS IMAGENS
    # ============================================================
    
    # Se você quiser processar várias imagens, use este código:
    """
    import glob
    
    # Listar todas as imagens em um diretório
    image_paths = glob.glob("sag/sag/*.nii.gz")
    
    print(f"\nEncontradas {len(image_paths)} imagens para processar")
    
    # Processar cada imagem
    all_results = []
    for img_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processando: {img_path}")
        print(f"{'='*60}")
        
        seg, desc, img = analyze_ventricle(
            image_path=img_path,
            threshold=20,
            save_images=True,
            output_dir='results'
        )
        
        if desc is not None:
            # Guardar resultados
            desc['image_path'] = img_path
            all_results.append(desc)
    
    # Salvar todos os resultados em um único CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv('results/all_descriptors.csv', index=False)
    print(f"\n✓ Todos os descritores salvos em: results/all_descriptors.csv")
    """