import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# 1. Load original data
df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")

# 2. Feature Selection & Scaling
features = [
    'MonthlyOrders', 'GrowthFactor', 'AOV', 
    'InStoreShare', 'UE_share', 'DD_share', 'SD_share', 
    'COGSRate', 'OPEXRate', 'CommissionRate', 
    'DeliveryCostPerOrder', 'DeliveryRadiusKM'
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 3. Dimensionality Reduction (PCA)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'], df['PCA3'] = pca_result[:, 0], pca_result[:, 1], pca_result[:, 2]

# 4. Unsupervised Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_mapping = {
    0: "Overextended, Low Return",
    1: "Aggregator-Dependent Low Margin",
    2: "Margin Fragile",
    3: "Scalable Self-Delivery Leaders",
    4: "Stable Local Performers"
}
df['ClusterLabel'] = df['Cluster'].map(cluster_mapping)

# 5. Growth Potential Index (GPI) Calculation
pct_growth = rankdata(df['MonthlyOrders'] * df['GrowthFactor']) / len(df)
pct_cost = 1 - (rankdata(df['COGSRate'] + df['OPEXRate']) / len(df))
pct_balance = 1 - (rankdata(df['UE_share'] + df['DD_share']) / len(df))
pct_logistics = rankdata(df['DeliveryRadiusKM'] / (df['DeliveryCostPerOrder'] + 1)) / len(df)

df['GPI'] = (0.35 * pct_growth + 0.25 * pct_cost + 0.20 * pct_balance + 0.20 * pct_logistics) * 100

# 6. Strategic Recommendations
def assign_recommendation(row):
    if row['GPI'] >= 75: return 'Expand / Scale'
    elif row['GPI'] >= 50 and row['UE_share'] + row['DD_share'] > 0.6: return 'Rebalance Channels'
    elif row['GPI'] >= 50: return 'Optimize Operations'
    else: return 'Hold / Stabilize'

df['StrategicRecommendation'] = df.apply(assign_recommendation, axis=1)

# Add Total Net Profit for the dashboard
df['TotalNetProfit'] = df['InStoreNetProfit'] + df['UberEatsNetProfit'] + df['DoorDashNetProfit'] + df['SelfDeliveryNetProfit']

# 7. Export the final engineered dataset
df.to_csv("Restaurant_Strategic_Classification.csv", index=False)
print("Success! Data engineered and saved as 'Restaurant_Strategic_Classification.csv'")