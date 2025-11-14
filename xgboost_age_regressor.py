"""
Regressor XGBoost para Predição de Idade - OASIS-2
Prevê a idade dos pacientes no momento do exame usando características morfológicas

Estrutura esperada dos CSVs:
- ventricle_descriptors_full.csv: Características morfológicas com MRI ID
- oasis_longitudinal_demographic.csv: Dados clínicos com MRI ID e Age
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import sys
from pathlib import Path

class BrainAgeRegressor:
    def __init__(self, model_path='modelo_xgboost_age.pkl', scaler_path='scaler_age.pkl'):
        """
        Inicializa o regressor de idade
        
        Args:
            model_path: Caminho para salvar/carregar o modelo
            scaler_path: Caminho para salvar/carregar o scaler
        """
        self.model_path = model_path 
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def train(self, features_csv='ventricle_descriptors_full.csv', 
              clinical_csv='oasis_longitudinal_demographic.csv',
              test_size=0.2, random_state=42):
        """
        Treina o modelo de regressão XGBoost para predizer idade
        Usa APENAS características morfológicas (não usa features clínicas)
        
        Args:
            features_csv: CSV com características morfológicas
            clinical_csv: CSV com dados clínicos (apenas para obter a idade - target)
            test_size: Proporção dos dados para teste
            random_state: Semente para reprodutibilidade
        """
        print("=" * 60)
        print("TREINAMENTO DO REGRESSOR DE IDADE - XGBOOST")
        print("=" * 60)
        
        # Características esperadas (apenas morfológicas)
        expected_features = ['area', 'circularity', 'eccentricity', 
                           'rectangularity', 'solidity', 'diameter']
        
        # Colunas a ignorar
        ignore_columns = ['MRI ID', 'image_path', 'Group']
        
        # 1. Carregar dados de características morfológicas
        print(f"\n1. Carregando dados de características de {features_csv}...")
        df_features = pd.read_csv(features_csv)
        print(f"   Dataset de características: {df_features.shape[0]} amostras, {df_features.shape[1]} colunas")
        
        # 2. Carregar dados clínicos APENAS para obter a idade (target)
        print(f"\n2. Carregando dados clínicos de {clinical_csv}...")
        df_clinical = pd.read_csv(clinical_csv, sep=';', decimal=',')
        print(f"   Dataset clínico: {df_clinical.shape[0]} amostras")
        print(f"   (Usando APENAS a coluna 'Age' como target)")
        
        # 3. Fazer merge dos datasets usando MRI ID
        print("\n3. Combinando datasets para obter o target (idade)...")
        df = pd.merge(df_features, df_clinical[['MRI ID', 'Age']], on='MRI ID', how='inner')
        print(f"   Dataset combinado: {df.shape[0]} amostras")
        
        # Verificar se há valores ausentes em Age
        missing_age = df['Age'].isna().sum()
        if missing_age > 0:
            print(f"   AVISO: {missing_age} amostras sem idade. Removendo...")
            df = df.dropna(subset=['Age'])
            print(f"   Dataset após remoção: {df.shape[0]} amostras")
        
        # 4. Separar features e target
        print("\n4. Preparando features e target...")
        print(f"   IMPORTANTE: Usando APENAS características morfológicas do ventrículo")
        
        # Target é a idade
        y = df['Age'].values
        
        # Features - remover Age e colunas de identificação
        X = df.drop(columns=['Age'])
        
        # Remover colunas de identificação
        cols_to_drop = [col for col in ignore_columns if col in X.columns]
        if cols_to_drop:
            print(f"   Removendo colunas de identificação: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
        
        # Selecionar apenas colunas numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        self.feature_names = list(X.columns)
        
        print(f"   Features utilizadas: {self.feature_names}")
        print(f"   Estatísticas da idade (target):")
        print(f"     Mínima: {y.min():.1f} anos")
        print(f"     Máxima: {y.max():.1f} anos")
        print(f"     Média: {y.mean():.1f} anos")
        print(f"     Desvio padrão: {y.std():.1f} anos")
        
        # 5. Dividir em treino e teste
        print(f"\n5. Dividindo dados (treino: {int((1-test_size)*100)}%, teste: {int(test_size*100)}%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"   Treino: {X_train.shape[0]} amostras")
        print(f"   Teste: {X_test.shape[0]} amostras")
        
        # 6. Normalizar dados
        print("\n6. Normalizando features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 7. Treinar modelo XGBoost para regressão
        print("\n7. Treinando modelo XGBoost Regressor...")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            objective='reg:squarederror'
        )
        self.model.fit(X_train_scaled, y_train)
        print("   Modelo treinado com sucesso!")
        
        # 8. Avaliar no conjunto de teste
        print("\n8. Avaliando modelo no conjunto de TESTE...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calcular erro percentual médio absoluto
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Exibir resultados
        print("\n" + "=" * 60)
        print("RESULTADOS DA AVALIAÇÃO (CONJUNTO DE TESTE)")
        print("=" * 60)
        print(f"\nMétricas de Regressão:")
        print(f"  R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} anos")
        print(f"  Mean Absolute Error (MAE): {mae:.4f} anos")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        print(f"\nInterpretação:")
        print(f"  - Em média, as predições erram por ±{mae:.1f} anos")
        print(f"  - O modelo explica {r2*100:.1f}% da variância da idade")
        
        # Análise de resíduos
        residuals = y_test - y_pred
        print(f"\nAnálise de Resíduos:")
        print(f"  Resíduo médio: {np.mean(residuals):.2f} anos")
        print(f"  Desvio padrão dos resíduos: {np.std(residuals):.2f} anos")
        
        # Mostrar alguns exemplos
        print("\n" + "-" * 60)
        print("EXEMPLOS DE PREDIÇÕES")
        print("-" * 60)
        print(f"{'Idade Real':>12} {'Idade Predita':>15} {'Erro':>10}")
        print("-" * 60)
        for i in range(min(10, len(y_test))):
            error = y_test[i] - y_pred[i]
            print(f"{y_test[i]:>12.1f} {y_pred[i]:>15.1f} {error:>10.1f}")
        
        # 9. Importância das features
        print("\n" + "-" * 60)
        print("IMPORTÂNCIA DAS CARACTERÍSTICAS")
        print("-" * 60)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.4f}")
        
        # 10. Salvar modelo e scaler
        print("\n10. Salvando modelo e scaler...")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"   Modelo salvo em: {self.model_path}")
        print(f"   Scaler salvo em: {self.scaler_path}")
        
        print("\n" + "=" * 60)
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'feature_importance': feature_importance
        }
    
    def load_model(self):
        """
        Carrega modelo e scaler salvos
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}. Execute o treinamento primeiro.")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado em {self.scaler_path}. Execute o treinamento primeiro.")
        
        print(f"Carregando modelo de {self.model_path}...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        print("Modelo carregado com sucesso!")
    
    def predict(self, features):
        """
        Faz predições de idade para novas entradas
        
        Args:
            features: DataFrame ou array com as características
            
        Returns:
            Array com as idades preditas
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        # Se for DataFrame, garantir que tem as features corretas
        if isinstance(features, pd.DataFrame):
            if self.feature_names:
                features = features[self.feature_names]
        
        # Normalizar e predizer
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def predict_from_csv(self, csv_path):
        """
        Faz predições de idade a partir de um CSV com características
        
        Args:
            csv_path: Caminho para CSV com características
            
        Returns:
            DataFrame com predições
        """
        print(f"\nCarregando dados de {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Remover colunas de identificação
        ignore_columns = ['MRI ID', 'image_path', 'Group', 'Age']
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
        
        predictions = self.predict(features)
        
        # Criar DataFrame com resultados
        results = pd.DataFrame({
            'MRI_ID': df['MRI ID'] if 'MRI ID' in df.columns else range(len(predictions)),
            'idade_predita': predictions,
        })
        
        # Se houver idade real, adicionar e calcular erro
        if 'Age' in df.columns:
            results['idade_real'] = df['Age']
            results['erro'] = results['idade_real'] - results['idade_predita']
            results['erro_abs'] = np.abs(results['erro'])
        
        print("\nPredições realizadas:")
        print(results)
        
        if 'idade_real' in results.columns:
            mae = results['erro_abs'].mean()
            print(f"\nErro Absoluto Médio: {mae:.2f} anos")
        
        return results


def main():
    """
    Função principal com menu interativo
    """
    regressor = BrainAgeRegressor()
    
    print("\n" + "=" * 60)
    print("REGRESSOR DE IDADE - XGBOOST")
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
        features_csv = input("\nCaminho do CSV com características (Enter para 'ventricle_descriptors_full.csv'): ").strip()
        if not features_csv:
            features_csv = 'ventricle_descriptors_full.csv'
        
        clinical_csv = input("Caminho do CSV com dados clínicos (Enter para 'oasis_longitudinal_demographic.csv'): ").strip()
        if not clinical_csv:
            clinical_csv = 'oasis_longitudinal_demographic.csv'
        
        regressor.train(features_csv=features_csv, clinical_csv=clinical_csv)
        
    elif mode == '2':
        # Usar modelo para predição
        try:
            regressor.load_model()
            
            csv_path = input("\nCaminho do CSV com características para predição: ").strip()
            results = regressor.predict_from_csv(csv_path)
            
            save_results = input("\nSalvar resultados em CSV? (s/n): ").strip().lower()
            if save_results == 's':
                output_path = input("Nome do arquivo de saída (Enter para 'predicoes_idade.csv'): ").strip()
                if not output_path:
                    output_path = 'predicoes_idade.csv'
                results.to_csv(output_path, index=False)
                print(f"Resultados salvos em {output_path}")
                
        except FileNotFoundError as e:
            print(f"\nERRO: {e}")
            print("Execute o treinamento primeiro (opção 1).")
    
    elif mode == '3':
        print("\nEncerrando...")
    else:
        print("\nOpção inválida!")


if __name__ == "__main__":
    main()