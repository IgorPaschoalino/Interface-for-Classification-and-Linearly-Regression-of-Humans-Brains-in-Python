"""
Classificador XGBoost para Imagens Cerebrais OASIS-2
Treina o modelo uma vez e permite reutilização para novas predições

Estrutura esperada do CSV:
- MRI ID: Identificador da imagem
- Group: Grupo/Classe (ex: Nondemented, Demented, Converted)
- area: Área da região de interesse
- circularity: Medida de quão circular é a forma
- eccentricity: Medida do alongamento da forma
- rectangularity: Quão bem a forma se ajusta a um retângulo
- solidity: Razão entre área e área convexa
- diameter: Diâmetro da região
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import os
import sys
from pathlib import Path

class BrainClassifier:
    def __init__(self, model_path='modelo_xgboost.pkl', scaler_path='scaler.pkl'):
        """
        Inicializa o classificador
        
        Args:
            model_path: Caminho para salvar/carregar o modelo
            scaler_path: Caminho para salvar/carregar o scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.label_encoder = None  # ADICIONADO
        self.feature_names = None
        
    def train(self, csv_path='ventricle_descriptors_full.csv', target_column='Group', 
              test_size=0.2, random_state=42):
        """
        Treina o modelo XGBoost
        
        Args:
            csv_path: Caminho para o arquivo CSV com características
            target_column: Nome da coluna com os rótulos
            test_size: Proporção dos dados para teste
            random_state: Semente para reprodutibilidade
        """
        print("=" * 60)
        print("TREINAMENTO DO MODELO XGBOOST")
        print("=" * 60)
        
        # Características esperadas (nomes exatos do CSV)
        expected_features = ['area', 'circularity', 'eccentricity', 
                           'rectangularity', 'solidity', 'diameter']
        
        # Colunas que devem ser ignoradas
        ignore_columns = ['MRI ID', 'image_path']
        
        # Carregar dados
        print(f"\n1. Carregando dados de {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"   Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
        print(f"   Colunas encontradas: {list(df.columns)}")
        
        # Verificar se a coluna target existe
        if target_column not in df.columns:
            print(f"\nERRO: Coluna '{target_column}' não encontrada no CSV.")
            print(f"Colunas disponíveis: {list(df.columns)}")
            return None
        
        # Separar features e target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # CODIFICAR AS CLASSES (STRINGS -> NÚMEROS)
        print(f"\n   Classes originais: {y.unique()}")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"   Classes codificadas: {np.unique(y_encoded)}")
        print(f"   Mapeamento: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Remover colunas de identificação que não são features
        cols_to_drop = [col for col in ignore_columns if col in X.columns]
        if cols_to_drop:
            print(f"   Removendo colunas de identificação: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
        
        # Selecionar apenas colunas numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        self.feature_names = list(X.columns)
        
        # Verificar se temos as características esperadas
        print(f"\n   Características esperadas: {expected_features}")
        missing_features = set(expected_features) - set(self.feature_names)
        if missing_features:
            print(f"   AVISO: Características faltando: {missing_features}")
        extra_features = set(self.feature_names) - set(expected_features)
        if extra_features:
            print(f"   Características extras encontradas: {extra_features}")
        
        print(f"   Features selecionadas: {len(self.feature_names)}")
        print(f"   Distribuição das classes:\n{pd.Series(y).value_counts()}")
        
        # Dividir em treino e teste
        print(f"\n2. Dividindo dados (treino: {int((1-test_size)*100)}%, teste: {int(test_size*100)}%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        print(f"   Treino: {X_train.shape[0]} amostras")
        print(f"   Teste: {X_test.shape[0]} amostras")
        
        # Normalizar dados
        print("\n3. Normalizando features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar modelo XGBoost
        print("\n4. Treinando modelo XGBoost...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss'
        )
        self.model.fit(X_train_scaled, y_train)
        print("   Modelo treinado com sucesso!")
        
        # Avaliar no conjunto de teste
        print("\n5. Avaliando modelo no conjunto de TESTE...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calcular sensibilidade e especificidade
        n_classes = len(np.unique(y_test))
        if n_classes == 2:  # Classificação binária
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:  # Classificação multiclasse
            sensitivity = np.diag(cm) / cm.sum(axis=1)
            specificity = []
            for i in range(len(cm)):
                tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                fp = cm[:, i].sum() - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificity.append(spec)
        
        # Exibir resultados
        print("\n" + "=" * 60)
        print("RESULTADOS DA AVALIAÇÃO (CONJUNTO DE TESTE)")
        print("=" * 60)
        print(f"\nAcurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if n_classes == 2:
            print(f"Sensibilidade (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
            print(f"Especificidade: {specificity:.4f} ({specificity*100:.2f}%)")
        else:
            print(f"\nSensibilidade por classe:")
            for i, sens in enumerate(sensitivity):
                class_name = self.label_encoder.classes_[i]
                print(f"  {class_name}: {sens:.4f} ({sens*100:.2f}%)")
            print(f"\nEspecificidade por classe:")
            for i, spec in enumerate(specificity):
                class_name = self.label_encoder.classes_[i]
                print(f"  {class_name}: {spec:.4f} ({spec*100:.2f}%)")
        
        print("\n" + "-" * 60)
        print("MATRIZ DE CONFUSÃO (TESTE)")
        print("-" * 60)
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(cm)
        
        print("\n" + "-" * 60)
        print("RELATÓRIO DE CLASSIFICAÇÃO")
        print("-" * 60)
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Salvar modelo, scaler e label encoder
        print("\n6. Salvando modelo, scaler e mapeamento de labels...")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        print(f"   Modelo salvo em: {self.model_path}")
        print(f"   Scaler salvo em: {self.scaler_path}")
        print(f"   Label encoder salvo em: label_encoder.pkl")
        
        print("\n" + "=" * 60)
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm
        }
    
    def load_model(self):
        """
        Carrega modelo, scaler e label encoder salvos
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}. Execute o treinamento primeiro.")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado em {self.scaler_path}. Execute o treinamento primeiro.")
        
        if not os.path.exists('label_encoder.pkl'):
            raise FileNotFoundError(f"Label encoder não encontrado. Execute o treinamento primeiro.")
        
        print(f"Carregando modelo de {self.model_path}...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load('label_encoder.pkl')
        print("Modelo carregado com sucesso!")
        print(f"Classes: {list(self.label_encoder.classes_)}")
    
    def predict(self, features):
        """
        Faz predições para novas entradas
        
        Args:
            features: DataFrame ou array com as características
            
        Returns:
            Predições (labels originais) e probabilidades
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        # Se for DataFrame, garantir que tem as features corretas
        if isinstance(features, pd.DataFrame):
            if self.feature_names:
                features = features[self.feature_names]
        
        # Normalizar e predizer
        features_scaled = self.scaler.transform(features)
        predictions_encoded = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Decodificar predições para labels originais
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def predict_from_csv(self, csv_path):
        """
        Faz predições a partir de um CSV com novas características
        
        Args:
            csv_path: Caminho para CSV com características
            
        Returns:
            DataFrame com predições
        """
        print(f"\nCarregando dados de {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Remover colunas de identificação
        ignore_columns = ['MRI ID', 'image_path', 'Group']
        features_df = df.drop(columns=[col for col in ignore_columns if col in df.columns], errors='ignore')
        
        # Selecionar apenas colunas numéricas
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features = features_df[numeric_cols]
        
        if self.feature_names:
            # Garantir que temos as mesmas features do treinamento
            missing_cols = set(self.feature_names) - set(features.columns)
            if missing_cols:
                print(f"AVISO: Colunas faltando: {missing_cols}")
            features = features[[col for col in self.feature_names if col in features.columns]]
        
        predictions, probabilities = self.predict(features)
        
        # Criar DataFrame com resultados
        results = pd.DataFrame({
            'MRI_ID': df['MRI ID'] if 'MRI ID' in df.columns else range(len(predictions)),
            'predicao': predictions,
        })
        
        # Adicionar probabilidades para cada classe
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]
        
        print("\nPredições realizadas:")
        print(results)
        
        return results


def main():
    """
    Função principal com menu interativo
    """
    classifier = BrainClassifier()
    
    print("\n" + "=" * 60)
    print("CLASSIFICADOR DE IMAGENS CEREBRAIS - XGBOOST")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nModos disponíveis:")
        print("1 - Treinar novo modelo")
        print("2 - Usar modelo existente para predição")
        print("3 - Sair")
        mode = input("\nEscolha uma opção (1-3): ")
    
    if mode == '1':
        # Treinar modelo
        csv_path = input("\nCaminho do CSV com características (Enter para 'ventricle_descriptors_full.csv'): ").strip()
        if not csv_path:
            csv_path = 'ventricle_descriptors_full.csv'
        
        target_col = input("Nome da coluna com os rótulos (Enter para 'Group'): ").strip()
        if not target_col:
            target_col = 'Group'
        
        classifier.train(csv_path=csv_path, target_column=target_col)
        
    elif mode == '2':
        # Usar modelo para predição
        try:
            classifier.load_model()
            
            print("\nOpções de entrada:")
            print("1 - CSV com características")
            print("2 - Inserir características manualmente")
            option = input("\nEscolha uma opção (1-2): ")
            
            if option == '1':
                csv_path = input("\nCaminho do CSV com novas características: ").strip()
                results = classifier.predict_from_csv(csv_path)
                
                save_results = input("\nSalvar resultados em CSV? (s/n): ").strip().lower()
                if save_results == 's':
                    output_path = input("Nome do arquivo de saída (Enter para 'predicoes.csv'): ").strip()
                    if not output_path:
                        output_path = 'predicoes.csv'
                    results.to_csv(output_path, index=False)
                    print(f"Resultados salvos em {output_path}")
            else:
                print("\nModo manual não implementado nesta versão.")
                print("Use um CSV com as características.")
                
        except FileNotFoundError as e:
            print(f"\nERRO: {e}")
            print("Execute o treinamento primeiro (opção 1).")
    
    elif mode == '3':
        print("\nEncerrando...")
    else:
        print("\nOpção inválida!")


if __name__ == "__main__":
    main()