print("=" * 60)
print("🌍 WEATHER ANALYSIS SYSTEM")
print("=" * 60)

# ====================
# PHASE 1: SETUP
# ====================
print("\n📦 PHASE 1: Setup - Importing libraries")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# Set style for better visualizations
plt.style.use('default')

# ====================
# PHASE 2: DATA ACQUISITION
# ====================
print("\n📥 PHASE 2: Data Acquisition")

# Create comprehensive dataset
data = {
    'City': ['New York', 'London', 'Tokyo', 'Sydney', 'Paris', 'Moscow', 
            'Berlin', 'Mumbai', 'Cairo', 'São Paulo', 'Toronto', 'Dubai',
            'Singapore', 'Beijing', 'Cape Town', 'Mexico City', 'Rome',
            'Seoul', 'Bangkok', 'Istanbul'],
    'Country': ['USA', 'UK', 'Japan', 'Australia', 'France', 'Russia',
               'Germany', 'India', 'Egypt', 'Brazil', 'Canada', 'UAE',
               'Singapore', 'China', 'South Africa', 'Mexico', 'Italy',
               'South Korea', 'Thailand', 'Turkey'],
    'Continent': ['North America', 'Europe', 'Asia', 'Australia', 'Europe',
                 'Europe', 'Europe', 'Asia', 'Africa', 'South America',
                 'North America', 'Asia', 'Asia', 'Asia', 'Africa',
                 'North America', 'Europe', 'Asia', 'Asia', 'Europe'],
    'Temperature': [15.5, 10.2, 16.8, 18.5, 12.3, 5.6, 8.9, 28.3, 22.1,
                   23.7, 2.4, 25.6, 30.2, 3.8, 19.7, 14.2, 13.5, 1.2,
                   32.5, 9.8],
    'Humidity': [65, 78, 72, 60, 75, 55, 70, 85, 40, 68, 62, 35, 88, 45,
                58, 52, 65, 48, 78, 66],
    'Rainfall': [120, 150, 180, 130, 110, 90, 95, 210, 5, 135, 80, 10,
                240, 25, 45, 70, 115, 40, 180, 105],
    'Wind_Speed': [12.3, 8.5, 6.2, 15.1, 10.4, 7.8, 9.3, 5.6, 14.2, 6.9,
                  11.5, 8.7, 3.4, 16.3, 13.8, 7.1, 5.9, 12.6, 4.8, 10.1],
    'Pressure': [1015, 1010, 1012, 1013, 1011, 1014, 1012, 1009, 1015,
                1010, 1013, 1016, 1008, 1017, 1014, 1012, 1013, 1015,
                1007, 1012],
    'Date': ['2024-01-15'] * 20,
    'Weather_Condition': ['Cloudy', 'Rainy', 'Partly Cloudy', 'Sunny',
                         'Cloudy', 'Snow', 'Rainy', 'Humid', 'Sunny',
                         'Cloudy', 'Snow', 'Clear', 'Thunderstorm',
                         'Windy', 'Sunny', 'Clear', 'Partly Cloudy',
                         'Cold', 'Hot', 'Rainy']
}

df = pd.DataFrame(data)
print("✅ Dataset created successfully!")
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"📋 Total Cities: {len(df)}")

print("\n🔍 First 5 rows of data:")
print(df.head())

# ====================
# PHASE 3: DATA CLEANING
# ====================
print("\n🧹 PHASE 3: Data Cleaning")

# Check for missing values
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("✅ No missing values found!")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add calculated columns
print("\n📊 Adding calculated columns...")

# Temperature categories
def categorize_temp(temp):
    if temp < 0:
        return 'Freezing'
    elif temp < 10:
        return 'Cold'
    elif temp < 20:
        return 'Cool'
    elif temp < 30:
        return 'Warm'
    else:
        return 'Hot'

df['Temp_Category'] = df['Temperature'].apply(categorize_temp)

# Rainfall intensity
def categorize_rainfall(rain):
    if rain < 50:
        return 'Low'
    elif rain < 100:
        return 'Moderate'
    elif rain < 200:
        return 'High'
    else:
        return 'Very High'

df['Rainfall_Intensity'] = df['Rainfall'].apply(categorize_rainfall)

# Comfort Index
df['Comfort_Index'] = 100 - (abs(df['Temperature'] - 22) + abs(df['Humidity'] - 50)/2)

print("✅ Added calculated columns!")

# ====================
# PHASE 4: DATA ANALYSIS
# ====================
print("\n📊 PHASE 4: Data Analysis")

# 1. Basic Statistics
print("\n1️⃣ BASIC STATISTICS:")
print("=" * 40)

# Temperature analysis
avg_temp = df['Temperature'].mean()
max_temp = df['Temperature'].max()
min_temp = df['Temperature'].min()
temp_range = max_temp - min_temp
hottest_city = df.loc[df['Temperature'].idxmax(), 'City']
coldest_city = df.loc[df['Temperature'].idxmin(), 'City']

print(f"🌡️  TEMPERATURE:")
print(f"   • Average: {avg_temp:.1f}°C")
print(f"   • Highest: {max_temp:.1f}°C in {hottest_city}")
print(f"   • Lowest: {min_temp:.1f}°C in {coldest_city}")
print(f"   • Range: {temp_range:.1f}°C")

# Rainfall analysis
avg_rain = df['Rainfall'].mean()
max_rain = df['Rainfall'].max()
rainy_city = df.loc[df['Rainfall'].idxmax(), 'City']

print(f"\n🌧️  RAINFALL:")
print(f"   • Average: {avg_rain:.1f}mm")
print(f"   • Highest: {max_rain:.1f}mm in {rainy_city}")

# 2. Categorical Analysis
print("\n2️⃣ CATEGORICAL ANALYSIS:")
print("=" * 40)

print("🌡️  Temperature Categories:")
temp_counts = df['Temp_Category'].value_counts()
for category, count in temp_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   • {category}: {count} cities ({percentage:.1f}%)")

print("\n🌧️  Rainfall Intensity:")
rain_counts = df['Rainfall_Intensity'].value_counts()
for intensity, count in rain_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   • {intensity}: {count} cities ({percentage:.1f}%)")

# 3. Continental Analysis
print("\n3️⃣ CONTINENTAL ANALYSIS:")
print("=" * 40)

continent_stats = df.groupby('Continent').agg({
    'Temperature': ['mean', 'min', 'max'],
    'Rainfall': 'mean',
    'Humidity': 'mean'
}).round(2)

print(continent_stats)

# 4. Top/Bottom Analysis
print("\n4️⃣ TOP & BOTTOM CITIES:")
print("=" * 40)

print("🏆 TOP 5 CITIES BY COMFORT INDEX:")
top_comfort = df.nlargest(5, 'Comfort_Index')[['City', 'Temperature', 'Humidity', 'Comfort_Index']]
print(top_comfort.to_string(index=False))

print("\n⚠️  BOTTOM 5 CITIES BY COMFORT INDEX:")
bottom_comfort = df.nsmallest(5, 'Comfort_Index')[['City', 'Temperature', 'Humidity', 'Comfort_Index']]
print(bottom_comfort.to_string(index=False))

print("\n☀️  SUNNIEST CITIES (Lowest Rainfall):")
sunny_cities = df.nsmallest(5, 'Rainfall')[['City', 'Rainfall', 'Weather_Condition']]
print(sunny_cities.to_string(index=False))

# ====================
# PHASE 5: VISUALIZATION
# ====================
print("\n🎨 PHASE 5: Data Visualization")

# Create figure with multiple subplots
fig = plt.figure(figsize=(15, 10))
fig.suptitle('🌍 Weather Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)

# 1. Temperature by City (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
colors = plt.cm.coolwarm((df['Temperature'] - df['Temperature'].min()) / 
                         (df['Temperature'].max() - df['Temperature'].min()))
bars = ax1.barh(df['City'], df['Temperature'], color=colors)
ax1.set_xlabel('Temperature (°C)')
ax1.set_title('Temperature by City', fontweight='bold')
ax1.axvline(x=avg_temp, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg_temp:.1f}°C')
ax1.legend()

# 2. Temperature Distribution (Histogram)
ax2 = plt.subplot(2, 3, 2)
ax2.hist(df['Temperature'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Frequency')
ax2.set_title('Temperature Distribution', fontweight='bold')
ax2.axvline(x=avg_temp, color='red', linestyle='--', label=f'Average: {avg_temp:.1f}°C')
ax2.legend()

# 3. Scatter: Temperature vs Rainfall
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(df['Temperature'], df['Rainfall'], 
                      c=df['Humidity'], s=df['Wind_Speed']*20, 
                      alpha=0.6, cmap='viridis')
ax3.set_xlabel('Temperature (°C)')
ax3.set_ylabel('Rainfall (mm)')
ax3.set_title('Temperature vs Rainfall', fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Humidity (%)')

# 4. Pie Chart: Temperature Categories
ax4 = plt.subplot(2, 3, 4)
temp_categories = df['Temp_Category'].value_counts()
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
wedges, texts, autotexts = ax4.pie(temp_categories.values, labels=temp_categories.index,
                                   autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax4.set_title('Temperature Categories', fontweight='bold')

# 5. Bar Plot: Average by Continent
ax5 = plt.subplot(2, 3, 5)
continent_avg = df.groupby('Continent')['Temperature'].mean().sort_values()
colors_cont = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FF99FF', '#FFD700']
bars5 = ax5.bar(continent_avg.index, continent_avg.values, color=colors_cont[:len(continent_avg)])
ax5.set_xlabel('Continent')
ax5.set_ylabel('Average Temperature (°C)')
ax5.set_title('Avg Temperature by Continent', fontweight='bold')
ax5.tick_params(axis='x', rotation=45)

# Add values on bars
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}°C', ha='center', va='bottom', fontsize=9)

# 6. Comfort Index Ranking
ax6 = plt.subplot(2, 3, 6)
df_sorted = df.sort_values('Comfort_Index', ascending=False)
colors_comfort = ['green' if x > 70 else 'orange' if x > 50 else 'red' 
                  for x in df_sorted['Comfort_Index']]
bars6 = ax6.barh(df_sorted['City'][:8], df_sorted['Comfort_Index'][:8], 
                 color=colors_comfort[:8])
ax6.set_xlabel('Comfort Index')
ax6.set_title('Top Cities by Comfort Index', fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the dashboard
plt.savefig('weather_dashboard.png', dpi=120, bbox_inches='tight')
print("✅ Dashboard saved as 'weather_dashboard.png'")

# Show the dashboard
plt.show()

# ====================
# PHASE 6: DATA EXPORT
# ====================
print("\n💾 PHASE 6: Data Export")

# Save enhanced dataset
df.to_csv('weather_data_enhanced.csv', index=False)
print("✅ Enhanced data saved as 'weather_data_enhanced.csv'")

# Save analysis summary
with open('weather_summary.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("WEATHER ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Cities Analyzed: {len(df)}\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Hottest City: {hottest_city} ({max_temp}°C)\n")
    f.write(f"Coldest City: {coldest_city} ({min_temp}°C)\n")
    f.write(f"Average Temperature: {avg_temp:.1f}°C\n")
    f.write(f"Average Rainfall: {avg_rain:.1f}mm\n\n")
    
    f.write("TOP 5 MOST COMFORTABLE CITIES:\n")
    f.write("-" * 30 + "\n")
    for idx, row in top_comfort.iterrows():
        f.write(f"{row['City']}: {row['Comfort_Index']:.1f}\n")

print("✅ Summary saved as 'weather_summary.txt'")

# ====================
# FINAL SUMMARY
# ====================
print("\n" + "=" * 60)
print("📋 PROJECT SUMMARY")
print("=" * 60)

print(f"""
✅ PROJECT COMPLETE!

🌍 DATASET:
• Total Cities: {len(df)}
• Date: {df['Date'].min().date()}

📊 ANALYSIS PERFORMED:
1. 📥 Data creation & preprocessing
2. 🧹 Data cleaning & feature engineering
3. 📈 Statistical analysis
4. 🌡️ Continental climate comparison
5. 🏆 Ranking & categorization
6. 🎨 6 different visualizations
7. 💾 Data export & reporting

📈 KEY FINDINGS:
🌡️  Hottest city: {hottest_city} ({max_temp}°C)
❄️  Coldest city: {coldest_city} ({min_temp}°C)
📊 Average temperature: {avg_temp:.1f}°C
🌧️  Average rainfall: {avg_rain:.1f}mm
🏆 Most comfortable: {top_comfort.iloc[0]['City']} 
   (Comfort Index: {top_comfort.iloc[0]['Comfort_Index']:.1f})

💾 FILES CREATED:
• weather_data_enhanced.csv - Enhanced dataset
• weather_dashboard.png - Visualization dashboard
• weather_summary.txt - Analysis report

📊 VISUALIZATIONS CREATED:
1. Temperature bar chart by city
2. Temperature distribution histogram
3. Temperature vs Rainfall scatter plot
4. Temperature categories pie chart
5. Continental averages bar chart
6. Comfort index ranking
""")

print("=" * 60)
print("🎉 Analysis complete! Check the generated files.")
print("=" * 60)