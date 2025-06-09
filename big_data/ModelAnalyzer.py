import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import r2_score, explained_variance_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelAnalyzer:
    def __init__(self, df, feature_scaler, target_scaler):
        self.df = df
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.best_model_results = None
        
    def analyze_correlations(self):
        print(" АНАЛІЗ КОРЕЛЯЦІЇ ОЗНАК (ПІРСОН)")
        print("=" * 20)
        
        correlation_matrix = self.df.corr(method='pearson')
        
        plt.figure(figsize=(12, 10))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   linewidths=0.5,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Кореляційна матриця ознак (Пірсон)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n Найсильніші кореляції:")
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if corr_val is None or np.isnan(corr_val):
                    continue
                else:
                    correlations.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_val
                    ))
        
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for i, (feat1, feat2, corr) in enumerate(correlations[:10]):
            print(f"{i+1:2d}. {feat1:<12} ↔ {feat2:<12}: {corr:6.3f}")
            
        return correlation_matrix
    
    def create_feature_distributions(self):
        print("\n РОЗПОДІЛ ОЗНАК")
        print("=" * 30)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, column in enumerate(self.df.columns):
            if i < len(axes):
                axes[i].hist(self.df[column].dropna(), bins=50, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'Розподіл {column}', fontweight='bold')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Частота')
                axes[i].grid(True, alpha=0.3)
        
        for i in range(len(self.df.columns), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.show()
    
    def analyze_best_model(self, model, X_test, y_test, model_name):
        print(f"\n ДЕТАЛЬНИЙ АНАЛІЗ НАЙКРАЩОЇ МОДЕЛІ: {model_name}")
        print("=" * 70)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(X_test_tensor).cpu().numpy()
        
        preds_inv = self.target_scaler.inverse_transform(preds)
        y_test_inv = self.target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        self.best_model_results = {
            'predictions': preds_inv.flatten(),
            'actual': y_test_inv.flatten(),
            'model_name': model_name
        }
        
        self.calculate_regression_metrics()
        
        self.create_regression_plots()
        
        self.calculate_classification_metrics()
        
        return self.best_model_results
    
    def calculate_regression_metrics(self):
        pred = self.best_model_results['predictions']
        actual = self.best_model_results['actual']
        
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        mae = np.mean(np.abs(pred - actual))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        r2 = r2_score(actual, pred)
        explained_var = explained_variance_score(actual, pred)
        
        print(" МЕТРИКИ РЕГРЕСІЇ:")
        print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
        print(f"MAE (Mean Absolute Error): {mae:.4f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"R² Score: {r2:.4f}")
        print(f"Explained Variance Score: {explained_var:.4f}")
    
    def create_regression_plots(self):
        pred = self.best_model_results['predictions']
        actual = self.best_model_results['actual']
        model_name = self.best_model_results['model_name']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].scatter(actual, pred, alpha=0.6, s=20)
        axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Реальні значення')
        axes[0, 0].set_ylabel('Прогнози')
        axes[0, 0].set_title(f'Графік розсіювання\n{model_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        r2 = r2_score(actual, pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                       fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        residuals = actual - pred
        axes[0, 1].scatter(pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Прогнози')
        axes[0, 1].set_ylabel('Залишки')
        axes[0, 1].set_title('Графік залишків')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Залишки')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].set_title('Розподіл залишків')
        axes[1, 0].grid(True, alpha=0.3)
        
        n_points = min(100, len(actual))
        time_idx = range(n_points)
        axes[1, 1].plot(time_idx, actual[-n_points:], label='Реальні', linewidth=2, alpha=0.8)
        axes[1, 1].plot(time_idx, pred[-n_points:], label='Прогнози', linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Час')
        axes[1, 1].set_ylabel('Ціна')
        axes[1, 1].set_title(f'Часовий ряд (останні {n_points} точок)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_classification_metrics(self):
        pred = self.best_model_results['predictions']
        actual = self.best_model_results['actual']
        
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(pred) > 0
        
        if len(actual_direction) == 0:
            print("Недостатньо даних для класифікаційних метрик")
            return
        
        print(f"\n КЛАСИФІКАЦІЙНІ МЕТРИКИ (Напрямок руху ціни):")
        print("Класи: 0 = Спадання, 1 = Зростання")
        
        accuracy = accuracy_score(actual_direction, pred_direction)
        precision, recall, f1, support = precision_recall_fscore_support(
            actual_direction, pred_direction, average='weighted'
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print(f"\n Детальний класифікаційний звіт:")
        report = classification_report(actual_direction, pred_direction, 
                                     target_names=['Спадання', 'Зростання'])
        print(report)
        
        self.create_confusion_matrix(actual_direction, pred_direction)
        
        self.create_comparison_table(accuracy, precision, recall, f1)
    
    def create_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Спадання', 'Зростання'],
                   yticklabels=['Спадання', 'Зростання'])
        plt.title('Матриця невідповідностей\n(Напрямок руху ціни)', fontsize=14, fontweight='bold')
        plt.xlabel('Прогнози')
        plt.ylabel('Реальні значення')
        plt.tight_layout()
        plt.show()
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\n Інтерпретація матриці невідповідностей:")
        print(f"True Negatives (правильно передбачені спадання): {tn}")
        print(f"False Positives (помилково передбачені зростання): {fp}")
        print(f"False Negatives (пропущені зростання): {fn}")
        print(f"True Positives (правильно передбачені зростання): {tp}")
    
    def create_comparison_table(self, accuracy, precision, recall, f1):
        print(f"\n ТАБЛИЦЯ ПОРІВНЯННЯ МЕТРИК:")
        print("=" * 50)
        
        metrics_data = {
            'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Значення': [accuracy, precision, recall, f1],
            'Опис': [
                'Частка правильних прогнозів',
                'Частка правильних позитивних прогнозів',
                'Частка знайдених позитивних випадків',
                'Гармонічне середнє Precision і Recall'
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        print(f"{'Метрика':<12} {'Значення':<10} {'Опис':<40}")
        print("-" * 65)
        for _, row in df_metrics.iterrows():
            print(f"{row['Метрика']:<12} {row['Значення']:<10.4f} {row['Опис']:<40}")
        
        plt.figure(figsize=(10, 6))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [accuracy, precision, recall, f1]
        
        bars = plt.bar(metrics_names, metrics_values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
        plt.title('Порівняння класифікаційних метрик', fontsize=14, fontweight='bold')
        plt.ylabel('Значення')
        plt.ylim(0, 1)
        
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_feature_importance_analysis(self, model, X_test, feature_names):
        print(f"\n АНАЛІЗ ВАЖЛИВОСТІ ОЗНАК")
        print("=" * 40)
        
        print(" Статистика параметрів моделі:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Загальна кількість параметрів: {total_params:,}")
        print(f"Навчальні параметри: {trainable_params:,}")
        
        feature_stats = pd.DataFrame({
            'Feature': feature_names,
            'Mean': np.mean(X_test, axis=0),
            'Std': np.std(X_test, axis=0),
            'Min': np.min(X_test, axis=0),
            'Max': np.max(X_test, axis=0)
        })
        
        print(f"\n Статистика вхідних ознак:")
        print(feature_stats.round(4))

def run_complete_analysis(df, feature_scaler, target_scaler, best_model, 
                         X_test, y_test, model_name):

    print(" ПОЧАТОК ПОВНОГО АНАЛІЗУ МОДЕЛІ")
    print("=" * 60)
    
    analyzer = ModelAnalyzer(df, feature_scaler, target_scaler)
    
    correlation_matrix = analyzer.analyze_correlations()
    
    analyzer.create_feature_distributions()
    
    results = analyzer.analyze_best_model(best_model, X_test, y_test, model_name)
    
    feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    analyzer.create_feature_importance_analysis(best_model, X_test, feature_names)
    
    print(f"\n АНАЛІЗ ЗАВЕРШЕНО!")
    print("=" * 60)
    
    return analyzer, results

