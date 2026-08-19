# Eficiência Académica: prever foco e desempenho no exame a partir de hábitos dos estudantes

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-8A2BE2?style=for-the-badge&logo=python&logoColor=white)

> 📖 Quick note in English: This README is also available in English. To access it, just click [here](README.md).

## Sobre o projeto

Este projeto analisa um dataset sintético com 5000 registos de estudantes (`Dados_Projeto/student_records_missing.csv`, com uma versão de referência completa em `student_records_full.csv`) para estudar a **eficiência académica** — se o desempenho no exame depende apenas das horas de estudo ou de uma combinação mais ampla de fatores como foco, descanso, hábitos digitais e bem-estar geral. Foi desenvolvido para a disciplina de **Fundamentos de Aprendizagem Automática** e define dois cenários preditivos complementares: um problema de **regressão** que estima o `focus_index` (proxy do nível de concentração durante o estudo) e um problema de **classificação** que prevê `high_exam_score`, uma variável binária construída a partir do quartil superior de `exam_score`.

O projeto foi desenvolvido pelo Grupo 1: Guilherme Soares, Duarte Soares e Vitória Correia.

### Funcionalidades
- Carregamento dos dados, construção do target (`high_exam_score`, derivado do quartil superior de `exam_score`) e inspeção exploratória das variáveis numéricas/categóricas e dos valores em falta
- Um pipeline de pré-processamento reprodutível (`Pipeline`/`ColumnTransformer`): imputação pela mediana + `RobustScaler` para variáveis numéricas, imputação pela categoria mais frequente + one-hot encoding para variáveis categóricas, ajustado apenas no conjunto de treino para evitar data leakage
- Modelos baseline — Regressão Linear e Árvore de Decisão para a tarefa de regressão, Regressão Logística e Árvore de Decisão para a tarefa de classificação — avaliados com divisão treino/teste e validação cruzada (`KFold`/`StratifiedKFold`)
- Avaliação dos modelos com RMSE/MAE para regressão e F1-score/ROC-AUC/accuracy para classificação, complementada com matrizes de confusão, gráficos de coeficientes e visualizações da árvore de decisão
- Análise de erro, inspecionando os maiores resíduos na regressão e os falsos positivos/negativos do classificador
- Engenharia e seleção de atributos: correlation filtering, variance threshold e seleção embedded baseada em importância (AdaBoost para classificação, Gradient Boosting para regressão), com procura por validação cruzada do número ótimo de atributos
- Modelos mais complexos (MLPClassifier, Random Forest, Gradient Boosting) com ajuste sistemático de hiperparâmetros via `RandomizedSearchCV`/`GridSearchCV`
- Aprendizagem não supervisionada: clustering com K-Means (com análise de elbow/silhouette) para identificar "personas" latentes de estudantes com base em atributos comportamentais e de bem-estar, sem recorrer às variáveis-alvo
- Interpretabilidade dos modelos com SHAP, comparando explicações locais entre Random Forest e Regressão Logística em casos de verdadeiro/falso positivo/negativo

### Tecnologias utilizadas
- Python (Jupyter Notebook)
- Pandas / NumPy
- scikit-learn (pipelines, modelos lineares, árvores, ensembles, MLP, clustering, seleção de modelos, seleção de atributos, métricas)
- Matplotlib / Seaborn
- SciPy (`loguniform` para distribuições de randomized search)
- SHAP
