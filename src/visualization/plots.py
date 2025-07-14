import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plot_drift(df1, df2, column, log=False, label1="Mes 1", label2="Mes 2", figsize=(5, 4)):
    """
    Gráfico de comparación simple
    """
    plt.figure(figsize=(10, 4))

    df1[column] = df1[column].dropna()
    df2[column] = df2[column].dropna()
    min1 = df1[column].min()
    min2 = df2[column].min()
    
    # Histogramas/Barras
    if log:
        df1[column] = df1[column].apply(lambda x: np.log1p(x - min1 + 1))
        df2[column] = df2[column].apply(lambda x: np.log1p(x - min2 + 1))
    
    plt.hist(df1[column].dropna(), bins=30, alpha=0.6, label=label1, density=True)
    plt.hist(df2[column].dropna(), bins=30, alpha=0.6, label=label2, density=True)
    plt.title(f'Distribuciones{" (log)" if log else ""}: {column}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_metrics_months(df, y_line, img_save_path, title='Métricas del Modelo por Mes', figsize=(14, 7), color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8):
    ax = df.plot(kind='bar', figsize=figsize, title=title, color=color, alpha=alpha)
    ax.set_xlabel('Mes')
    ax.set_ylabel('Score')
    ax.set_ylim(0, df.max().max()*1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Líneas de referencia
    ax.axhline(y=y_line, color=color[1], linestyle='--', alpha=alpha)

    plt.savefig(img_save_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()


def plot_labels_months(df_size, df_predicts, img_save_path, title='Métricas del Modelo por Mes', figsize=(20, 5), color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8):
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    df_size.plot(kind='bar', ax=ax[0], color=color[0], alpha=alpha)
    ax[0].set_xlabel('Mes')
    ax[0].set_ylabel('Size')
    ax[0].legend(['Size'])
    ax[0].grid(True, alpha=0.3, axis='y')
    ax[0].set_xticklabels(df_size.index, rotation=45)

    df_predicts.plot(kind='bar', ax=ax[1], color=color[1:], alpha=alpha)
    ax[1].set_xlabel('Mes')
    ax[1].set_ylabel('Ratio')
    ax[1].legend(['Media y', 'Media y predicha'])
    ax[1].grid(True, alpha=0.3, axis='y')
    ax[1].set_xticklabels(df_predicts.index, rotation=45)

    plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()