import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Carregamento e Preparação dos Dados
# (Assumindo que os dados já foram limpos na Fase 1)
df = pd.read_csv('seu_dataset.csv')

# Identificar as variáveis-alvo a excluir conforme o ficheiro ideia.txt
targets = ['focus_index', 'exam_score']
features = df.drop(columns=targets)

# Normalização: Essencial para Clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 2. Determinação do número ideal de clusters (Heurísticas)
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

# Visualização do Elbow Method e Silhouette Score
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(k_range, inertia, 'bo-', label='Inertia (Elbow)')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inertia', color='b')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')

plt.title('Determinação do Número Ideal de Clusters')
plt.show()

# 3. Treino do Modelo Final
# (Exemplo: k=3 baseado na análise visual dos gráficos acima)
optimal_k = 3 
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(features_scaled)

# 4. Definição das Personas
# Agrupar pela média para identificar características dominantes
cluster_profiles = df.groupby('cluster').mean()

print("\n--- Perfis Médios por Cluster ---")
print(cluster_profiles)

# Visualização comparativa das Personas (Heatmap das médias)
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profiles.drop(columns=targets, errors='ignore'), annot=True, cmap='YlGnBu')
plt.title('Persona Analysis: Médias por Cluster')
plt.show()

def describe_personas(profiles, global_mean):
    """
    Gera uma descrição textual comparando cada cluster com a média global
    para ajudar a definir a "Persona".
    """
    print("\n=== Análise de Personas (Destaques em relação à média global) ===")
    for cluster_id in profiles.index:
        print(f"\nCluster {cluster_id}:")
        diff = profiles.loc[cluster_id] - global_mean
        
        # Identificar os 2 traços mais fortes (acima ou abaixo da média)
        top_traits = diff.sort_values(ascending=False)
        
        high_traits = top_traits.index[top_traits > 0][:2].tolist()
        low_traits = top_traits.index[top_traits < 0][-2:].tolist()
        
        print(f" - Características Fortes: {', '.join(high_traits)}")
        print(f" - Abaixo da Média: {', '.join(low_traits)}")
        
        # Sugestão de Nome de Persona baseada em lógica simples
        if 'study_hours' in high_traits and 'sleep_quality' in low_traits:
            print(" > Sugestão de Nome: O Estudante Esforçado mas Exausto")
        elif 'digital_balance' in high_traits:
            print(" > Sugestão de Nome: O Estudante Equilibrado Digitalmente")
        elif 'focus_index' in targets and df[df['cluster'] == cluster_id]['focus_index'].mean() > df['focus_index'].mean():
            print(" > Sugestão de Nome: O Estudante de Alta Performance")
        else:
            print(" > Sugestão de Nome: Perfil Intermédio")

# Cálculo da média global para comparação
global_mean = df.drop(columns=['cluster']).mean()
describe_personas(cluster_profiles, global_mean)
