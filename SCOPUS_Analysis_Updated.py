# ============================================
# IMPORTS
# ============================================
import requests
import pandas as pd
import time
from collections import defaultdict
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib_venn import venn2, venn2_circles
import numpy as np
import seaborn as sb
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Load existing data (1920-2021)
# ============================================
df = pd.read_csv('merged_article_counts_by_country_across_years.csv')
print(f"Loaded: {len(df)} countries, years 1920-2021")

# ============================================
# Fetch article counts for 2022-2025 & update totals to 1920-2025
# ============================================
api_key = 'eb2cec89a26c7449245d9379a4f9944e'
base_url = 'https://api.elsevier.com/content/search/scopus'
headers = {'Accept': 'application/json'}

# Countries where Scopus uses a different name than the backbone
scopus_name_map = {
    'United States of America': 'United States',
    'Hong Kong S.A.R.': 'Hong Kong',
    'Vietnam': 'Viet Nam',
    'Syria': 'Syrian Arab Republic',
    'Ivory Coast': "Cote d'Ivoire",
    'Democratic Republic of the Congo': 'Democratic Republic Congo',
    'Russia': 'Russian Federation',
}

for i in range(len(df)):
    country = df.loc[i, 'country']
    if pd.isnull(country):
        continue

    query_country = scopus_name_map.get(country, country)

    print(f"[{i+1}/{len(df)}] {country}...", end=" ", flush=True)

    for year in range(2022, 2026):
        query = f'TITLE-ABS-KEY("anesthes*" OR "anaesthes*") AND AFFILCOUNTRY({query_country})'
        params = {'query': query, 'count': 25, 'date': str(year), 'apiKey': api_key}
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            total = data.get('search-results', {}).get('opensearch:totalResults', 0)
            df.loc[i, str(year)] = int(total)
        time.sleep(1)

    # Update total count (1920-2025)
    query = f'TITLE-ABS-KEY("anesthes*" OR "anaesthes*") AND AFFILCOUNTRY({query_country})'
    params = {'query': query, 'count': 25, 'date': '1920-2025', 'apiKey': api_key}
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        total = data.get('search-results', {}).get('opensearch:totalResults', 0)
        df.loc[i, 'anesthesiology'] = int(total)

    print("done.")

# ============================================
# Update HDI to 2023 values from 2025 HDR
# ============================================
hdi_new = pd.read_csv('backbone_anesthesiology_2023hdi.csv')
hdi_lookup = hdi_new.set_index('country')[['hdi_2023', 'hdicode']].to_dict('index')

for i, row in df.iterrows():
    if row['country'] in hdi_lookup:
        info = hdi_lookup[row['country']]
        df.loc[i, 'hdi_2023'] = info['hdi_2023']
        df.loc[i, 'hdicode'] = info['hdicode']

if 'hdi_2022' in df.columns:
    df = df.drop(columns=['hdi_2022'])

# ============================================
# Save updated data
# ============================================
df.to_csv('UPDATED_merged_article_counts_by_country_across_years.csv', index=False)
print(f"Saved! Shape: {df.shape}")
print(df[['country', 'hdicode', 'hdi_2023', 'anesthesiology', '2022', '2023', '2024', '2025']].head(10))

# ============================================
# Adjudication of results. Script written for Sierra Leone.
# ============================================
api_key = 'eb2cec89a26c7449245d9379a4f9944e'
base_url = 'https://api.elsevier.com/content/search/scopus'
headers = {'Accept': 'application/json'}
query = 'TITLE-ABS-KEY("anesthes*" OR "anaesthes*") AND AFFILCOUNTRY("Sierra Leone")'
params = {
    'query': query,
    'count': 25,
    'date': '1920-2025',
    'apiKey': api_key
}

all_entries = []
page_number = 1
while True:
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        total = data.get('search-results', {})
        entries = total.get('entry', [])
        for entry in tqdm(entries, desc=f"Processing Page {page_number}", leave=False):
            title = entry.get('dc:title', "No Title")
            authors = entry.get('dc:creator', "No Author")
            publication_name = entry.get('prism:publicationName', "No Publication Name")
            publication_date = entry.get('prism:coverDate', "No Date")
            all_entries.append({
                "title": title,
                "authors": authors,
                "publication_name": publication_name,
                "publication_date": publication_date
            })
        links = total.get('link', [])
        next_link = None
        for link in links:
            if link.get('@ref') == 'next':
                next_link = link.get('@href')
                break
        if next_link:
            parsed_url = urlparse(next_link)
            params['start'] = parse_qs(parsed_url.query).get('start', [None])[0]
            page_number += 1
        else:
            break
    else:
        print(f"Error: {response.status_code}")
        break

adj_df = pd.DataFrame(all_entries)
print(f"Collected {len(adj_df)} entries.")
print(adj_df.head())
adj_df.to_csv('script_adjudication_sierra_leone.csv')

# ============================================
# Creating World Dataset from Total Article Counts
# ============================================
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)
data_input = pd.read_csv('UPDATED_merged_article_counts_by_country_across_years.csv')
oecd = pd.read_csv('oecd_iso3_codes.csv')
world = world.merge(data_input, left_on='ADM0_A3', right_on='iso3', how='left')
world['log'] = np.log10(world['anesthesiology'].abs()+1)

# ============================================
# Atlas Generation on World Dataset
# ============================================
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5,)
plot = world.plot(column='log', ax=ax, legend=True,
           legend_kwds={'label': "Log10 Normalized National Publication Volume",
                        'orientation': "horizontal"},
           cmap='Blues').set_axis_off()
plt.savefig("National Publication Density.pdf", format="pdf")
plt.show()

# ============================================
# Analysis on World dataset (Violin Plot)
# ============================================
sb.set(style = 'white')
desired_order = ['Very High', 'High', 'Medium', 'Low']
sb.violinplot(x ="hdicode", y = np.log10(data_input['anesthesiology'].abs()+1),
              order = desired_order, palette = "PuBu_r",
              data = data_input)
plt.savefig("HDI Distribution.pdf", format="pdf")
plt.show()

# ============================================
# Bland-Altman Plot
# ============================================
f, ax = plt.subplots(1, figsize = (8,5))
sm.graphics.mean_diff_plot(np.log10(data_input['anesthesiology'].abs()+1), np.log10(data_input['total_yearly'].abs()+1), ax = ax)
ax.set_ylim(-1, 1)
plt.savefig("Bland-Altman.pdf", format="pdf")
plt.show()

# ============================================
# Statistical Analyses
# ============================================
print("\n=== HDI Category Counts ===")
print(data_input['hdicode'].value_counts())

hdi_very_high = data_input.loc[data_input["hdicode"]=="Very High", ]
hdi_high = data_input.loc[data_input["hdicode"]=="High", ]
hdi_medium = data_input.loc[data_input["hdicode"]=="Medium", ]
hdi_low = data_input.loc[data_input["hdicode"]=="Low", ]
top_performers = data_input.sort_values(by='anesthesiology', ascending=False).head(10)
oecd_performers = oecd.merge(data_input, left_on='ISO3', right_on='iso3', how='left').drop(['Country','ISO3'], axis=1)

stat, p_value = kruskal(np.log10(hdi_very_high['anesthesiology'].abs()+1),
                        np.log10(hdi_high['anesthesiology'].abs()+1),
                        np.log10(hdi_medium['anesthesiology'].abs()+1),
                        np.log10(hdi_low['anesthesiology'].abs()+1))

print("\n=== Kruskal-Wallis Test ===")
print("Kruskal-Wallis statistic:", stat)
print("p-value:", p_value)

print("\n=== Top 10 Performers ===")
print(top_performers[['country', 'anesthesiology', 'hdicode']].to_string(index=False))

# ============================================
# Venn Diagram of Top Performers and OECD
# ============================================
set1 = set(top_performers['iso3'])
set2 = set(oecd_performers['iso3'])

fig, ax = plt.subplots(figsize=(8, 6))
venn = venn2([set1, set2], set_labels=('Top Performers', 'OECD'), ax=ax, set_colors=('#f7fbff', '#08306b'), alpha=0.7)

venn2_circles([set1, set2], linestyle='solid', linewidth=1.5, color='black')

for text in venn.set_labels:
    text.set_fontsize(14)
    text.set_fontweight('bold')

for text in venn.subset_labels:
    if text:
        text.set_fontsize(12)

plt.title('Overlap Between Top Performers and OECD', fontsize=16, fontweight='bold')

for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("Venn.pdf", format="pdf")

plt.tight_layout()
plt.show()
only_top = sorted(set1 - set2)
only_oecd = sorted(set2 - set1)
overlap = sorted(set1 & set2)

venn_table = pd.DataFrame({
    'Only in Top Performers': pd.Series(only_top),
    'Overlap': pd.Series(overlap),
    'Only in OECD': pd.Series(only_oecd)
})

print("\n=== Venn Diagram Table ===")
print(venn_table)

# ============================================
# Scatterplot
# ============================================
sb.scatterplot(data=data_input, x=np.log10(data_input['anesthesiology'].abs()+1), y=np.log10(data_input['total_yearly'].abs()+1))
corr_coef, p_value = pearsonr(np.log10(data_input['anesthesiology'].abs()+1), np.log10(data_input['total_yearly'].abs()+1))
m, b = np.polyfit(np.log10(data_input['anesthesiology'].abs()+1), np.log10(data_input['total_yearly'].abs()+1), 1)
plt.plot(np.log10(data_input['anesthesiology'].abs()+1), m*np.array(np.log10(data_input['anesthesiology'].abs()+1)) + b, color='black')
plt.title(f'Scatter plot\nCorrelation: {corr_coef:.2f}, p-value: {p_value:.3f}')
plt.savefig("Scatterplot.pdf", format="pdf")
plt.show()

print("\n=== Scatterplot Correlation ===")
print(f"Pearson correlation: {corr_coef:.4f}")
print(f"p-value: {p_value:.6f}")

# Outlier detection
data_input['residual'] = np.log10(data_input['total_yearly'].abs()+1) - (m * np.log10(data_input['anesthesiology'].abs()+1) + b)

print("\n=== Biggest Outliers Above the Line ===")
print(data_input.nlargest(5, 'residual')[['country', 'anesthesiology', 'total_yearly', 'residual']].to_string(index=False))

print("\n=== Biggest Outliers Below the Line ===")
print(data_input.nsmallest(5, 'residual')[['country', 'anesthesiology', 'total_yearly', 'residual']].to_string(index=False))

# ============================================
# Time Trend for Total Article Counts
# ============================================
start_year = '1920'
end_year = '2025'

df_range = data_input.loc[:, start_year:end_year]
column_sums = df_range.sum(axis=0)
plt.figure(figsize=(10, 6))
plt.plot(column_sums.index, column_sums.values, color='navy', label='Sum of Values')

plt.fill_between(column_sums.index, column_sums.values, color='steelblue', alpha=0.4)

plt.title('Sum of Columns over Time')
plt.xlabel('Year')
plt.ylabel('Sum of Values')
plt.grid(False)

xticks = [year for i, year in enumerate(column_sums.index) if i % 10 == 0]

plt.xticks(xticks)

plt.savefig("Time Series.pdf", format="pdf")
plt.show()

# ============================================
# Time Trend for Total Article Counts by HDI
# ============================================
start_year = '1960'
end_year = '2025'
plt.figure(figsize=(10, 6))

hdi_codes = ['Very High', 'High', 'Medium', 'Low']

colors = plt.cm.jet(np.linspace(0, 1, len(hdi_codes)))

for i, code in enumerate(hdi_codes):
    df_range = data_input.loc[data_input["hdicode"]==code, start_year:end_year]
    column_sums = df_range.sum(axis=0)
    plt.plot(column_sums.index, column_sums.values, label=code, color=colors[i])
    plt.fill_between(column_sums.index, column_sums.values, color=colors[i], alpha=0.4)

plt.title('Sum of Columns over Time')
plt.xlabel('Year')
plt.ylabel('Sum of Values')
plt.grid(False)

xticks = [year for i, year in enumerate(column_sums.index) if i % 10 == 0]

plt.xticks(xticks)

plt.legend(title='HDI Code')

plt.savefig("Time Series by HDI Code.pdf", format="pdf")
plt.show()

# ============================================
# Time Trend for Total Article Counts in Top Performers
# ============================================
start_year = '1960'
end_year = '2025'
plt.figure(figsize=(10, 6))

nations = top_performers["country"]

colors = plt.cm.turbo(np.linspace(0, 1, len(top_performers['country'])))

for i, nation in enumerate(nations):
    df_range = top_performers.loc[top_performers["country"]==nation, start_year:end_year]
    column_sums = df_range.sum(axis=0)
    plt.plot(column_sums.index, column_sums.values, label=nation, color=colors[i])

plt.title('Sum of Columns over Time')
plt.xlabel('Year')
plt.ylabel('Sum of Values')
plt.grid(False)

xticks = [year for i, year in enumerate(column_sums.index) if i % 10 == 0]

plt.xticks(xticks)

plt.legend(title='Country')

plt.savefig("Time Series by Country (Top Performers).pdf", format="pdf")
plt.show()

# ============================================
# Long format transformation for all article counts
# ============================================
data_input_clean = data_input.drop(['hdicode','anesthesiology','total_yearly','residual'], axis=1)
data_input_clean_long = pd.melt(data_input_clean, id_vars=['country','hdi_2023','iso3'], var_name='year', value_name='count')
data_input_clean_long['year'] = data_input_clean_long['year'].astype(int)
data_input_final = data_input_clean_long[data_input_clean_long['year']>=1960].reset_index(drop=True)

# ============================================
# Long format transformation for top performer article counts
# ============================================
top_performers_clean = top_performers.drop(['hdicode','anesthesiology','total_yearly','hdi_2023','residual'], axis=1, errors='ignore')
top_performers_clean_long = pd.melt(top_performers_clean, id_vars=['country','iso3'], var_name='year', value_name='count')
top_performers_clean_long['year'] = top_performers_clean_long['year'].astype(int)
top_performers_final = top_performers_clean_long[top_performers_clean_long['year']>=1960].reset_index(drop=True)

# ============================================
# Long format transformation for OECD article counts
# ============================================
oecd_clean = oecd_performers.drop(['hdicode','anesthesiology','total_yearly','hdi_2023'], axis=1)
oecd_clean_long = pd.melt(oecd_clean, id_vars=['country','iso3'], var_name='year', value_name='count')
oecd_clean_long['year'] = oecd_clean_long['year'].astype(int)
oecd_final = oecd_clean_long[oecd_clean_long['year']>=1995].reset_index(drop=True)

# ============================================
# Long format transformation of World Bank indicators
# ============================================
world_bank = pd.read_csv('world_bank.csv')
world_bank_long = world_bank.melt(id_vars=['Series Code','Country Name','Country Code'], var_name='year', value_name='count')
world_bank_long = world_bank_long.drop('Country Name', axis=1)
world_bank_long['year'] = pd.to_numeric(world_bank_long['year'])
world_bank_long_pivot = world_bank_long.pivot_table(
    index=['Country Code', 'year'],
    columns='Series Code',
    values='count',
    aggfunc='sum'
).reset_index()
world_bank_long_pivot = world_bank_long_pivot.rename(columns={'Country Code': 'iso3'})
world_bank_long_pivot = world_bank_long_pivot.drop('GB.XPD.RSDV.GD.ZS', axis=1, errors='ignore')

# ============================================
# Merging Top Performer Article Counts with World Bank Indicators
# ============================================
pvar_dataset_top = top_performers_final.merge(world_bank_long_pivot, on = ['iso3','year'], how = 'left')
pvar_dataset_top['country'] = pvar_dataset_top['country'].astype(str)
pvar_dataset_top['year'] = pvar_dataset_top['year'].astype(int)
for col in ['count', 'NY.GDP.MKTP.CD','NY.GDP.PCAP.CD']:
    pvar_dataset_top[col] = pd.to_numeric(pvar_dataset_top[col], errors='coerce')
pvar_dataset_nona = pvar_dataset_top.dropna()

plt.figure(figsize=(10, 6))
scatter = sb.scatterplot(
    data=pvar_dataset_nona,
    x='year',
    y='count',
    hue='country',
    size='NY.GDP.MKTP.CD',
    sizes=(50, 500),
    palette=colors.tolist(),
    edgecolor='black',
    linewidth=0.8,
    legend=False
)

color_handles = [
    Line2D([0], [0], marker='o', color='w', label=country,
           markerfacecolor=color, markersize=10, markeredgecolor='black')
    for country, color in zip(pvar_dataset_nona['country'], sb.color_palette(colors.tolist()))
]

legend1 = plt.legend(handles=color_handles, title='Country', loc='upper left')
plt.gca().add_artist(legend1)
plt.savefig("Time Series by Country with GDP (Top Performers).pdf", format="pdf")
plt.show()

# ============================================
# Merging OECD Article Counts with World Bank Indicators
# ============================================
pvar_dataset_oecd = oecd_final.merge(world_bank_long_pivot, on = ['iso3','year'], how = 'left')
pvar_dataset_oecd['country'] = pvar_dataset_oecd['country'].astype(str)
pvar_dataset_oecd['year'] = pvar_dataset_oecd['year'].astype(int)
for col in ['count', 'NY.GDP.MKTP.CD','NY.GDP.PCAP.CD']:
    pvar_dataset_oecd[col] = pd.to_numeric(pvar_dataset_oecd[col], errors='coerce')
pvar_dataset_oecd_nona = pvar_dataset_oecd.dropna()

# ============================================
# Summary of all results
# ============================================
print("\n" + "="*50)
print("SUMMARY OF ALL RESULTS")
print("="*50)

print(f"\nTotal countries in dataset: {len(data_input)}")
print(f"\nHDI Category Distribution:")
print(data_input['hdicode'].value_counts().to_string())

stat, p_value = kruskal(np.log10(hdi_very_high['anesthesiology'].abs()+1),
                        np.log10(hdi_high['anesthesiology'].abs()+1),
                        np.log10(hdi_medium['anesthesiology'].abs()+1),
                        np.log10(hdi_low['anesthesiology'].abs()+1))
print(f"\nKruskal-Wallis Test:")
print(f"  Statistic: {stat:.4f}")
print(f"  p-value: {p_value:.6f}")

corr_coef, p_corr = pearsonr(np.log10(data_input['anesthesiology'].abs()+1), np.log10(data_input['total_yearly'].abs()+1))
print(f"\nPearson Correlation (anesthesiology vs total_yearly):")
print(f"  Correlation: {corr_coef:.4f}")
print(f"  p-value: {p_corr:.6f}")

print(f"\nTop 10 Performers:")
print(top_performers[['country', 'iso3', 'anesthesiology', 'hdicode', 'hdi_2023']].to_string(index=False))

print(f"\nVenn Diagram:")
print(f"  Only Top Performers: {only_top}")
print(f"  Overlap: {overlap}")
print(f"  Only OECD: {only_oecd}")

print(f"\nScatterplot Outlier:")
print(f"  Biggest outlier above line: {data_input.nlargest(1, 'residual')[['country', 'anesthesiology', 'total_yearly', 'residual']].to_string(index=False)}")

print("\n=== FILES SAVED ===")
print("PDFs: National Publication Density, HDI Distribution, Bland-Altman,")
print("  Scatterplot, Venn, Time Series, Time Series by HDI Code,")
print("  Time Series by Country (Top Performers),")
print("  Time Series by Country with GDP (Top Performers)")
print("CSVs: UPDATED_merged_article_counts_by_country_across_years.csv,")
print("  script_adjudication_sierra_leone.csv")
print("\nAll done!")
