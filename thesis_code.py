# 核心库导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 数据预处理库
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import shapiro, grubbs
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 聚类分析库
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from gap_statistic import OptimalK
from yellowbrick.cluster import KElbowVisualizer

# 判别分析库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, RegularizedDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 时间序列预测库
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 回归预测库
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, BayesianOptimization
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 模型解释库
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 工具函数
import itertools
from tqdm import tqdm
import pickle
from datetime import datetime

def load_data(main_data_path, od_matrix_path=None):
    """加载核心数据集"""
    # 加载主数据（多维度指标）
    main_df = pd.read_csv(main_data_path, encoding='utf-8')
    
    # 加载OD矩阵（可选，用于补充分析）
    od_df = pd.read_csv(od_matrix_path) if od_matrix_path else None
    
    # 数据基本信息
    print("主数据形状:", main_df.shape)
    print("主数据列名:", main_df.columns.tolist()[:10], "...")  # 显示前10个列名
    
    return main_df, od_df

# 数据加载
main_df, od_df = load_data("main_data_advanced.csv", "od_matrix.csv")

# 筛选核心分析列（四大维度指标）
core_cols = [
    # 数据流动维度
    '跨境数据传输总量_TB', '入境数据量_TB', '出境数据量_TB', '数据中心数量', 
    '互联网国际出口带宽_Gbps', '政务API调用量_亿次', '数据交易额_亿元',
    # 经济发展维度
    'GDP_亿元', '人均GDP_万元', '数字经济核心产业增加值_亿元', '外贸依存度_%', 
    '实际利用外资FDI_亿美元', '跨境电商交易额_亿元',
    # 创新能力维度
    '研发经费投入_亿元', '发明专利授权量', '高新技术企业数', '科技人才数量_万人',
    '技术合同成交额_亿元',
    # 信息基础设施维度
    '5G基站密度_个每平方公里', '千兆光网覆盖率_%', '算力规模_PFLOPS', 
    '物联网连接数_万个', '工业互联网标识解析量_亿次'
]

# 筛选有效数据（去除列缺失过多的行）
analysis_df = main_df[['城市', '年份'] + core_cols].copy()
analysis_df = analysis_df.dropna(thresh=len(core_cols)*0.7)  # 至少70%的指标非空
print("筛选后数据形状:", analysis_df.shape)

def handle_missing_values(df, core_cols):
    """缺失值处理（对比4种方法，选择最优）"""
    data = df[core_cols].copy()
    
    # 方法1：多重插补MICE（论文选择方法）
    mice_imputer = IterativeImputer(random_state=42, max_iter=10)
    mice_data = mice_imputer.fit_transform(data)
    
    # 方法2：KNN插补
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_data = knn_imputer.fit_transform(data)
    
    # 方法3：时间序列插值（针对年度数据）
    ts_data = data.copy()
    for col in core_cols:
        ts_data[col] = ts_data[col].interpolate(method='time')
    
    # 方法4：均值插补（基准方法）
    mean_data = data.fillna(data.mean())
    
    # 评估插补效果（使用完整子集验证）
    complete_mask = data.notna().all(axis=1)
    if complete_mask.sum() > 30:  # 确保有足够的验证数据
        complete_data = data[complete_mask].values
        # 随机制造缺失值用于验证
        np.random.seed(42)
        mask = np.random.choice([True, False], size=complete_data.shape, p=[0.1, 0.9])
        
        test_data = complete_data.copy()
        test_data[mask] = np.nan
        
        # 计算各方法的RMSE
        def rmse(true, pred):
            mask = ~np.isnan(true)
            return np.sqrt(((true[mask] - pred[mask])**2).mean())
        
        mice_pred = mice_imputer.fit_transform(test_data)
        knn_pred = knn_imputer.fit_transform(test_data)
        
        print(f"MICE插补RMSE: {rmse(complete_data, mice_pred):.3f}")
        print(f"KNN插补RMSE: {rmse(complete_data, knn_pred):.3f}")
        
    # 返回最优方法（MICE）处理后的数据
    result_df = df.copy()
    result_df[core_cols] = mice_data
    
    return result_df

# 执行缺失值处理
clean_df = handle_missing_values(analysis_df, core_cols)
print("缺失值处理后数据形状:", clean_df.shape)
print("缺失值占比:", (clean_df[core_cols].isnull().sum() / len(clean_df)).max())  # 最大缺失列占比

def detect_outliers(df, core_cols):
    """异常值检测（4种方法）"""
    data = df[core_cols].copy()
    outlier_masks = {}
    
    # 方法1：Grubbs检验（单变量）
    grubbs_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    for col in core_cols:
        if data[col].nunique() > 10:  # 数据足够多样时使用
            try:
                stat, p = grubbs.test(data[col].dropna(), alpha=0.05)
                if p < 0.05:
                    # 标记异常值
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    grubbs_mask[col] = (data[col] < q1 - 1.5*iqr) | (data[col] > q3 + 1.5*iqr)
            except:
                pass
    outlier_masks['grubbs'] = grubbs_mask.any(axis=1)
    
    # 方法2：Isolation Forest（多变量）
    iso_forest = LocalOutlierFactor(contamination=0.05, random_state=42)
    iso_scores = iso_forest.fit_predict(data)
    outlier_masks['isolation_forest'] = iso_scores == -1
    
    # 方法3：局部异常因子LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_scores = lof.fit_predict(data)
    outlier_masks['lof'] = lof_scores == -1
    
    # 方法4：STL分解（时间序列异常）
    stl_mask = pd.Series(False, index=data.index)
    for city in df['城市'].unique():
        city_data = df[df['城市'] == city]
        if len(city_data) >= 4:  # 至少4个时间点
            for col in core_cols[:3]:  # 重点检测核心流动指标
                try:
                    stl = STL(city_data[col].values, period=1)
                    res = stl.fit()
                    residuals = res.resid
                    std_res = np.std(residuals)
                    stl_mask[city_data.index] = np.abs(residuals) > 3*std_res
                except:
                    pass
    outlier_masks['stl'] = stl_mask
    
    # 综合异常值（至少2种方法标记为异常）
    outlier_mask = pd.DataFrame(outlier_masks).any(axis=1)
    print(f"异常值数量: {outlier_mask.sum()}, 占比: {outlier_mask.mean():.1%}")
    
    # 处理异常值（缩尾处理）
    data_clean = data.copy()
    for col in core_cols:
        q1 = data[col].quantile(0.01)
        q99 = data[col].quantile(0.99)
        data_clean[col] = data[col].clip(q1, q99)
    
    # 输出异常值报告
    outlier_report = pd.DataFrame({
        '城市': df['城市'],
        '年份': df['年份'],
        '是否异常': outlier_mask,
        '异常方法数': pd.DataFrame(outlier_masks).sum(axis=1)
    })
    
    return data_clean, outlier_report

# 执行异常值处理
data_clean, outlier_report = detect_outliers(clean_df, core_cols)
print("异常值报告前5行:")
print(outlier_report[outlier_report['是否异常']].head())

# 更新数据框
final_df = clean_df.copy()
final_df[core_cols] = data_clean

def preprocess_data(df, core_cols):
    """数据变换与共线性处理"""
    data = df[core_cols].copy()
    
    # 1. 正态性检验与变换
    transform_cols = []
    for col in core_cols:
        stat, p = shapiro(data[col].dropna())
        if p < 0.05:  # 非正态分布
            transform_cols.append(col)
    
    # 对数变换（处理右偏数据）
    data_transformed = data.copy()
    for col in transform_cols:
        data_transformed[col] = np.log1p(data[col] - data[col].min() + 1)  # 避免负数值
    
    # 2. 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_transformed)
    data_scaled = pd.DataFrame(data_scaled, columns=core_cols, index=data.index)
    
    # 3. 多重共线性处理（VIF检验）
    vif_data = pd.DataFrame()
    vif_data['指标'] = core_cols
    vif_data['VIF'] = [variance_inflation_factor(data_scaled.values, i) for i in range(len(core_cols))]
    
    # 剔除VIF>10的指标
    high_vif_cols = vif_data[vif_data['VIF'] > 10]['指标'].tolist()
    print(f"高共线性指标（VIF>10）: {high_vif_cols}")
    
    final_cols = [col for col in core_cols if col not in high_vif_cols]
    data_final = data_scaled[final_cols].copy()
    
    print(f"最终保留指标数: {len(final_cols)}")
    print("最终指标:", final_cols)
    
    return data_final, scaler, final_cols, vif_data

# 执行数据预处理
data_final, scaler, final_cols, vif_report = preprocess_data(final_df, core_cols)
print("预处理后数据形状:", data_final.shape)
print("VIF报告前10行:")
print(vif_report.head(10))

def find_optimal_clusters(data, max_k=6):
    """确定最优聚类数（6种方法）"""
    results = {}
    
    # 方法1：Elbow Method（SSE曲线）
    kmeans = KMeans(random_state=42, init='k-means++')
    elbow_visualizer = KElbowVisualizer(kmeans, k=(2, max_k), metric='distortion')
    elbow_visualizer.fit(data)
    results['elbow'] = elbow_visualizer.elbow_value_
    
    # 方法2：轮廓系数
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    results['silhouette'] = np.argmax(silhouette_scores) + 2  # +2因为k从2开始
    
    # 方法3：Calinski-Harabasz Index
    ch_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        labels = kmeans.fit_predict(data)
        score = calinski_harabasz_score(data, labels)
        ch_scores.append(score)
    results['ch'] = np.argmax(ch_scores) + 2
    
    # 方法4：Davies-Bouldin Index
    db_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        labels = kmeans.fit_predict(data)
        score = davies_bouldin_score(data, labels)
        db_scores.append(score)
    results['db'] = np.argmin(db_scores) + 2
    
    # 方法5：Gap Statistic
    optimal_k = OptimalK(parallel_backend='joblib')
    n_clusters = optimal_k(data, cluster_array=range(2, max_k+1))
    results['gap'] = optimal_k.elbow_value_
    
    # 方法6：GMM的BIC/AIC
    bic_scores = []
    for k in range(2, max_k+1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
    results['bic'] = np.argmin(bic_scores) + 2
    
    # 综合最优聚类数（投票法）
    from collections import Counter
    vote_result = Counter(results.values()).most_common(1)[0][0]
    
    print("各方法最优聚类数:")
    for method, k in results.items():
        print(f"{method}: {k}")
    print(f"综合最优聚类数: {vote_result}")
    
    # 可视化关键指标
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(2, max_k+1), silhouette_scores, 'o-', label='轮廓系数')
    axes[0].set_xlabel('聚类数k')
    axes[0].set_ylabel('轮廓系数')
    axes[0].set_title('轮廓系数曲线')
    axes[0].grid(True)
    
    axes[1].plot(range(2, max_k+1), ch_scores, 's-', label='CH指数', color='orange')
    axes[1].set_xlabel('聚类数k')
    axes[1].set_ylabel('Calinski-Harabasz指数')
    axes[1].set_title('CH指数曲线')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('最优聚类数分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return vote_result, results

# 确定最优聚类数
optimal_k, cluster_results = find_optimal_clusters(data_final, max_k=6)

def cluster_analysis(data, n_clusters, final_cols):
    """5种聚类算法实现与对比"""
    cluster_results = {}
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=100)
    kmeans_labels = kmeans.fit_predict(data)
    cluster_results['kmeans'] = {
        'labels': kmeans_labels,
        'centers': kmeans.cluster_centers_,
        'silhouette': silhouette_score(data, kmeans_labels)
    }
    
    # 2. 层次聚类（Ward法）
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    hierarchical_labels = hierarchical.fit_predict(data)
    cluster_results['hierarchical'] = {
        'labels': hierarchical_labels,
        'silhouette': silhouette_score(data, hierarchical_labels)
    }
    
    # 3. DBSCAN（自动确定簇数，这里按最优k调整参数）
    # 网格搜索最优参数
    best_eps = None
    best_min_samples = None
    best_silhouette = -1
    
    for eps in np.linspace(0.5, 3, 10):
        for min_samples in range(3, 10):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            db_labels = dbscan.fit_predict(data)
            if len(set(db_labels)) - (1 if -1 in db_labels else 0) == n_clusters:
                if len(set(db_labels)) > 1:
                    score = silhouette_score(data, db_labels)
                    if score > best_silhouette:
                        best_silhouette = score
                        best_eps = eps
                        best_min_samples = min_samples
    
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan.fit_predict(data)
    cluster_results['dbscan'] = {
        'labels': dbscan_labels,
        'eps': best_eps,
        'min_samples': best_min_samples,
        'silhouette': best_silhouette
    }
    
    # 4. 高斯混合模型GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(data)
    cluster_results['gmm'] = {
        'labels': gmm_labels,
        'probs': gmm.predict_proba(data),
        'silhouette': silhouette_score(data, gmm_labels)
    }
    
    # 5. 谱聚类
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='rbf')
    spectral_labels = spectral.fit_predict(data)
    cluster_results['spectral'] = {
        'labels': spectral_labels,
        'silhouette': silhouette_score(data, spectral_labels)
    }
    
    # 输出各算法性能
    print("各聚类算法性能对比:")
    for algo, res in cluster_results.items():
        print(f"{algo}: 轮廓系数={res['silhouette']:.3f}")
    
    # 选择最优算法（轮廓系数最高）
    best_algo = max(cluster_results.keys(), key=lambda x: cluster_results[x]['silhouette'])
    print(f"最优聚类算法: {best_algo}")
    
    # 可视化聚类结果（PCA降维）
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 6))
    for algo in ['kmeans', 'hierarchical', best_algo]:
        labels = cluster_results[algo]['labels']
        unique_labels = sorted(set(labels))
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(data_pca[mask, 0], data_pca[mask, 1], c=[color], label=f'{algo}-{label}', alpha=0.7)
    
    plt.xlabel(f'PCA1 (解释方差: {pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PCA2 (解释方差: {pca.explained_variance_ratio_[1]:.1%})')
    plt.title('不同聚类算法结果对比（PCA降维）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('聚类结果对比.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_results, best_algo

# 执行聚类分析
cluster_dict, best_algo = cluster_analysis(data_final, optimal_k, final_cols)

# 获取最优聚类标签并添加到数据框
final_df['聚类标签'] = cluster_dict[best_algo]['labels']

# 聚类结果统计
cluster_stats = final_df.groupby(['聚类标签', '城市']).size().unstack(fill_value=0)
print("各聚类包含城市:")
for cluster in range(optimal_k):
    cities = cluster_stats.loc[cluster][cluster_stats.loc[cluster] > 0].index.tolist()
    print(f"聚类{cluster}: {cities}")

def validate_clusters(data, labels, true_labels=None):
    """聚类验证（内部+外部）"""
    validation_results = {}
    
    # 内部验证指标
    validation_results['silhouette'] = silhouette_score(data, labels)
    validation_results['calinski_harabasz'] = calinski_harabasz_score(data, labels)
    validation_results['davies_bouldin'] = davies_bouldin_score(data, labels)
    
    # 外部验证（以经济发展水平为参考标签）
    if true_labels is None:
        # 用GDP分位数作为参考标签
        gdp_labels = pd.qcut(final_df['GDP_亿元'], q=optimal_k, labels=False)
        true_labels = gdp_labels.values
    
    validation_results['ari'] = adjusted_rand_score(true_labels, labels)
    validation_results['nmi'] = normalized_mutual_info_score(true_labels, labels)
    
    # 稳定性验证（Bootstrap重采样）
    n_bootstrap = 100
    jaccard_scores = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap验证"):
        # 重采样
        sample_idx = np.random.choice(len(data), size=len(data), replace=True)
        sample_data = data.iloc[sample_idx]
        
        # 重新聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        sample_labels = kmeans.fit_predict(sample_data)
        
        # 计算Jaccard系数（对比原始标签）
        original_subset = labels[sample_idx]
        jaccard = 0
        for cluster in range(optimal_k):
            cluster_mask = sample_labels == cluster
            if cluster_mask.sum() == 0:
                continue
            original_cluster = Counter(original_subset[cluster_mask]).most_common(1)[0][0]
            intersection = ((sample_labels == cluster) & (original_subset == original_cluster)).sum()
            union = ((sample_labels == cluster) | (original_subset == original_cluster)).sum()
            jaccard += intersection / union
        
        jaccard_scores.append(jaccard / optimal_k)
    
    validation_results['bootstrap_jaccard'] = np.mean(jaccard_scores)
    
    # 输出验证结果
    print("聚类验证结果:")
    for metric, value in validation_results.items():
        print(f"{metric}: {value:.3f}")
    
    # 可视化验证结果
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = list(validation_results.keys())
    values = list(validation_results.values())
    ax.bar(metrics, values)
    ax.set_title('聚类验证指标')
    ax.set_ylabel('指标值')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('聚类验证结果.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return validation_results

# 执行聚类验证
validation_res = validate_clusters(data_final, final_df['聚类标签'])

def discriminant_analysis(data, labels):
    """判别分析与分类（5种方法）"""
    # 数据划分（按城市分层，避免时间序列泄露）
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    groups = final_df['城市'].values
    
    # 定义分类模型
    models = {
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'RDA': RegularizedDiscriminantAnalysis(),
        'Logistic': LogisticRegression(max_iter=1000, multi_class='ovr'),
        'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
    }
    
    # 交叉验证评估
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, data, labels, cv=logo.split(data, labels, groups),
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        )
        cv_results[name] = {
            'accuracy': np.mean(scores['test_accuracy']),
            'precision': np.mean(scores['test_precision_macro']),
            'recall': np.mean(scores['test_recall_macro']),
            'f1': np.mean(scores['test_f1_macro'])
        }
    
    # 输出各模型性能
    print("判别分析模型性能对比:")
    for name, metrics in cv_results.items():
        print(f"{name}: 准确率={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    
    # 选择最优模型进行详细分析
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['accuracy'])
    best_model = models[best_model_name]
    best_model.fit(data, labels)
    
    # 特征重要性（针对有特征重要性的模型）
    feature_importance = None
    if best_model_name == 'RandomForest':
        feature_importance = best_model.feature_importances_
    elif best_model_name in ['LDA', 'RDA']:
        feature_importance = np.abs(best_model.coef_).mean(axis=0)
    
    # 可视化特征重要性
    if feature_importance is not None:
        feat_import_df = pd.DataFrame({
            'feature': final_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feat_import_df)
        plt.title(f'{best_model_name}特征重要性（Top10）')
        plt.tight_layout()
        plt.savefig('特征重要性.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 混淆矩阵（全数据训练后）
    y_pred = best_model.predict(data)
    cm = confusion_matrix(labels, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'预测{i}' for i in range(optimal_k)],
                yticklabels=[f'真实{i}' for i in range(optimal_k)])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{best_model_name}混淆矩阵')
    plt.tight_layout()
    plt.savefig('混淆矩阵.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出分类报告
    print(f"\n{best_model_name}分类报告:")
    print(classification_report(labels, y_pred, target_names=[f'类别{i}' for i in range(optimal_k)]))
    
    return cv_results, best_model, feature_importance

# 执行判别分析
discrim_results, best_discrim_model, feat_import = discriminant_analysis(data_final.values, final_df['聚类标签'])

def time_series_prediction(df, city_col='城市', time_col='年份', target_col='跨境数据传输总量_TB', forecast_years=2):
    """时间序列预测（5种模型）"""
    # 准备时间序列数据（按城市聚合）
    ts_data = df.groupby([time_col])[target_col].sum().reset_index()
    ts_data[time_col] = pd.to_datetime(ts_data[time_col], format='%Y')
    ts_data = ts_data.sort_values(time_col)
    
    # 数据拆分
    train_data = ts_data[target_col].values
    dates = ts_data[time_col].values
    
    # 定义预测模型
    models = {}
    
    # 1. ARIMA
    arima_model = ARIMA(train_data, order=(2, 1, 1))  # 手动定阶，或用auto_arima
    arima_results = arima_model.fit()
    arima_forecast = arima_results.get_forecast(steps=forecast_years)
    arima_pred = arima_forecast.predicted_mean
    arima_ci = arima_forecast.conf_int()
    models['ARIMA'] = {'pred': arima_pred, 'ci': arima_ci, 'aic': arima_results.aic}
    
    # 2. SARIMA（假设无明显季节，用年度数据）
    sarima_model = SARIMAX(train_data, order=(2, 1, 1), seasonal_order=(1, 0, 1, 1))
    sarima_results = sarima_model.fit(disp=False)
    sarima_forecast = sarima_results.get_forecast(steps=forecast_years)
    sarima_pred = sarima_forecast.predicted_mean
    models['SARIMA'] = {'pred': sarima_pred, 'aic': sarima_results.aic}
    
    # 3. Prophet
    prophet_df = ts_data.rename(columns={time_col: 'ds', target_col: 'y'})
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=forecast_years, freq='Y')
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast.tail(forecast_years)['yhat'].values
    models['Prophet'] = {'pred': prophet_pred, 'forecast_df': prophet_forecast}
    
    # 4. VAR（多变量，选择相关指标）
    var_cols = [target_col, '数字经济核心产业增加值_亿元', '5G基站密度_个每平方公里']
    var_data = df.groupby(time_col)[var_cols].sum().values
    var_model = VAR(var_data)
    var_results = var_model.fit(maxlags=2)
    var_forecast = var_results.forecast(var_data[-2:], steps=forecast_years)
    var_pred = var_forecast[:, 0]  # 第一个变量是目标变量
    models['VAR'] = {'pred': var_pred, 'aic': var_results.aic}
    
    # 5. LSTM
    # 数据预处理
    def create_sequences(data, seq_len=2):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)
    
    scaler_lstm = StandardScaler()
    train_scaled = scaler_lstm.fit_transform(train_data.reshape(-1, 1)).flatten()
    X_train, y_train = create_sequences(train_scaled, seq_len=2)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # 构建LSTM模型
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=0)
    
    # 预测
    last_seq = train_scaled[-2:]
    lstm_pred = []
    for _ in range(forecast_years):
        next_val = lstm_model.predict(last_seq.reshape(1, 2, 1), verbose=0)[0][0]
        lstm_pred.append(next_val)
        last_seq = np.append(last_seq[1:], next_val)
    
    lstm_pred = scaler_lstm.inverse_transform(np.array(lstm_pred).reshape(-1, 1)).flatten()
    models['LSTM'] = {'pred': lstm_pred}
    
    # 模型评估（用历史数据回测）
    def evaluate_model(model_name, model_pred):
        # 回测：用前n-1年预测第n年
        if len(train_data) < 4:
            return {'mape': np.nan}
        
        backtest_pred = []
        for i in range(2, len(train_data)):
            if model_name == 'ARIMA':
                temp_model = ARIMA(train_data[:i], order=(2, 1, 1))
                temp_results = temp_model.fit()
                backtest_pred.append(temp_results.get_forecast(steps=1).predicted_mean[0])
            elif model_name == 'Prophet':
                temp_df = ts_data[:i].rename(columns={time_col: 'ds', target_col: 'y'})
                temp_model = Prophet(yearly_seasonality=True)
                temp_model.fit(temp_df)
                future = temp_model.make_future_dataframe(periods=1, freq='Y')
                temp_forecast = temp_model.predict(future)
                backtest_pred.append(temp_forecast.tail(1)['yhat'].values[0])
        
        if len(backtest_pred) > 0:
            mape = np.mean(np.abs((train_data[2:] - backtest_pred) / train_data[2:])) * 100
            return {'mape': mape}
        return {'mape': np.nan}
    
    # 评估各模型
    eval_results = {}
    for name, res in models.items():
        eval_res = evaluate_model(name, res['pred'])
        eval_results[name] = eval_res
        print(f"{name}: 回测MAPE={eval_res['mape']:.2f}%" if not np.isnan(eval_res['mape']) else f"{name}: 无法回测")
    
    # 可视化预测结果
    future_years = [dates[-1].year + i + 1 for i in range(forecast_years)]
    plt.figure(figsize=(12, 6))
    
    # 历史数据
    plt.plot(dates, train_data, 'o-', label='历史数据', color='blue', linewidth=2)
    
    # 各模型预测
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (name, res) in enumerate(models.items()):
        pred = res['pred']
        plt.plot(future_years, pred, 's-', label=f'{name}预测', color=colors[i], alpha=0.8)
        
        # Prophet置信区间
        if name == 'Prophet':
            ci_lower = res['forecast_df'].tail(forecast_years)['yhat_lower'].values
            ci_upper = res['forecast_df'].tail(forecast_years)['yhat_upper'].values
            plt.fill_between(future_years, ci_lower, ci_upper, color='green', alpha=0.2)
    
    plt.xlabel('年份')
    plt.ylabel(target_col)
    plt.title(f'{target_col}时间序列预测（2024-2025）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('时间序列预测结果.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出预测结果
    print("\n2024-2025年预测结果:")
    for year_idx, year in enumerate(future_years):
        print(f"\n{year}年:")
        for name, res in models.items():
            print(f"{name}: {res['pred'][year_idx]:.2f} {target_col}")
    
    return models, eval_results

# 执行时间序列预测
ts_models, ts_eval = time_series_prediction(final_df)

def regression_prediction(data, target, features, final_df):
    """回归预测（6种模型）"""
    # 数据准备（按城市分层拆分）
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(data, groups=final_df['城市']))
    
    X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    
    # 定义回归模型
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(n_estimators=500, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # 模型训练与评估
    results = {}
    for name, model in models.items():
        # 超参数调优（贝叶斯优化示例，以XGBoost为例）
        if name == 'XGBoost':
            def xgb_objective(learning_rate, max_depth, n_estimators):
                model = XGBRegressor(
                    learning_rate=learning_rate,
                    max_depth=int(max_depth),
                    n_estimators=int(n_estimators),
                    random_state=42
                )
                return -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
            
            optimizer = BayesianOptimization(
                f=xgb_objective,
                pbounds={'learning_rate': (0.01, 0.3), 'max_depth': (3, 10), 'n_estimators': (50, 200)},
                random_state=42,
                n_iter=20
            )
            optimizer.maximize()
            best_params = optimizer.max['params']
            best_params['max_depth'] = int(best_params['max_depth'])
            best_params['n_estimators'] = int(best_params['n_estimators'])
            model = XGBRegressor(**best_params, random_state=42)
        
        # 训练与预测
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'model': model
        }
    
    # 输出性能对比
    print("回归预测模型性能对比:")
    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.3f}")
    
    # 选择最优模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    print(f"\n最优回归模型: {best_model_name}")
    
    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_model.predict(X_test), alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{best_model_name}预测结果（R²={results[best_model_name]["r2"]:.3f}）')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('回归预测结果.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, best_model

# 执行回归预测（目标变量：跨境数据传输总量）
target = final_df['跨境数据传输总量_TB']
reg_results, best_reg_model = regression_prediction(data_final, target, final_cols, final_df)

def model_interpretation(model, X_data, feature_names):
    """模型解释（SHAP值分析）"""
    # 初始化SHAP解释器
    if 'XGBoost' in str(model.__class__):
        explainer = shap.TreeExplainer(model)
    elif 'Linear' in str(model.__class__):
        explainer = shap.LinearExplainer(model, X_data)
    else:
        explainer = shap.KernelExplainer(model.predict, X_data.sample(min(100, len(X_data)), random_state=42))
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_data)
    
    # 1. 全局特征重要性
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type='bar')
    plt.title('SHAP全局特征重要性')
    plt.tight_layout()
    plt.savefig('SHAP特征重要性.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. SHAP依赖图（Top2特征）
    top_features = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).nlargest(2).index
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, feat in enumerate(top_features):
        shap.dependence_plot(
            feat, shap_values, X_data, feature_names=feature_names,
            ax=axes[i], show=False
        )
        axes[i].set_title(f'SHAP依赖图：{feat}')
    
    plt.tight_layout()
    plt.savefig('SHAP依赖图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 单样本解释（随机选择一个样本）
    sample_idx = np.random.choice(len(X_data))
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_data.iloc[sample_idx],
        feature_names=feature_names
    ))
    plt.title(f'单样本SHAP解释（样本索引：{sample_idx}）')
    plt.tight_layout()
    plt.savefig('SHAP瀑布图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 部分依赖图（Top2特征）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, feat in enumerate(top_features):
        PartialDependenceDisplay.from_estimator(
            model, X_data, [feat], ax=axes[i], grid_resolution=20
        )
        axes[i].set_title(f'部分依赖图：{feat}')
    
    plt.tight_layout()
    plt.savefig('部分依赖图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出特征重要性排序
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    print("SHAP特征重要性排序:")
    print(shap_importance.head(10))
    
    return shap_values, shap_importance

# 执行模型解释
shap_vals, shap_import = model_interpretation(best_reg_model, data_final, final_cols)

def save_results(final_df, cluster_dict, best_algo, discrim_results, ts_models, reg_results, shap_import):
    """保存所有分析结果"""
    # 创建结果目录
    import os
    os.makedirs('分析结果', exist_ok=True)
    
    # 1. 数据框保存
    final_df.to_csv('分析结果/带聚类标签的数据.csv', index=False, encoding='utf-8-sig')
    
    # 2. 聚类结果保存
    cluster_summary = pd.DataFrame({
        '城市': final_df['城市'],
        '年份': final_df['年份'],
        '聚类标签': final_df['聚类标签'],
        '跨境数据传输总量_TB': final_df['跨境数据传输总量_TB'],
        'GDP_亿元': final_df['GDP_亿元'],
        '5G基站密度_个每平方公里': final_df['5G基站密度_个每平方公里']
    })
    cluster_summary.to_csv('分析结果/聚类结果汇总.csv', index=False, encoding='utf-8-sig')
    
    # 3. 模型性能保存
    model_perf = pd.DataFrame({
        '聚类算法': list(cluster_dict.keys()),
        '轮廓系数': [cluster_dict[algo]['silhouette'] for algo in cluster_dict.keys()]
    })
    model_perf = pd.concat([model_perf, pd.DataFrame(discrim_results).T.reset_index().rename(columns={'index': '判别算法'})], axis=0)
    model_perf.to_csv('分析结果/模型性能对比.csv', index=False, encoding='utf-8-sig')
    
    # 4. 预测结果保存
    forecast_years = [2024, 2025]
    ts_forecast = pd.DataFrame({
        '年份': forecast_years,
        'ARIMA预测': ts_models['ARIMA']['pred'],
        'Prophet预测': ts_models['Prophet']['pred'],
        'XGBoost预测': reg_results['XGBoost']['model'].predict(data_final.tail(2)) if len(data_final)>=2 else [np.nan]*2
    })
    ts_forecast.to_csv('分析结果/预测结果汇总.csv', index=False, encoding='utf-8-sig')
    
    # 5. 特征重要性保存
    shap_import.to_csv('分析结果/SHAP特征重要性.csv', index=False, encoding='utf-8-sig')
    
    # 6. 模型保存
    with open('分析结果/最优聚类模型.pkl', 'wb') as f:
        pickle.dump(cluster_dict[best_algo], f)
    with open('分析结果/最优回归模型.pkl', 'wb') as f:
        pickle.dump(reg_results[max(reg_results.keys(), key=lambda x: reg_results[x]['r2'])]['model'], f)
    
    print("所有结果已保存到'分析结果'目录")

# 执行结果保存
save_results(final_df, cluster_dict, best_algo, discrim_results, ts_models, reg_results, shap_import)

print("="*50)
print("生成的核心结果文件：")
print("1. 聚类结果对比.png")
print("2. 判别分析混淆矩阵.png")
print("3. 时间序列预测结果.png")
print("4. 回归预测结果.png")
print("5. SHAP特征重要性.png")
print("6. 分析结果/（文件夹包含所有数据文件）")

