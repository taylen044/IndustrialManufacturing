import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title='Industrial Manufacturing | Stock Screener', page_icon='🏭', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.stApp{background:#f3f0ec;color:#18243e}h1,h2,h3{color:#18243e!important}div[data-testid="stMetric"]{background:#fff;border:1px solid #ddd6d0;border-radius:12px;padding:14px 18px}section[data-testid="stSidebar"]{background:#19243b}section[data-testid="stSidebar"] *{color:#f8fafc!important}section[data-testid="stSidebar"] div[data-baseweb="select"] *{color:#18243e!important}section[data-testid="stSidebar"] input{color:#18243e!important}.eyebrow{font-size:13px;font-weight:800;letter-spacing:3px;color:#b85a2d}.subtle{color:#536078;font-size:15px}.panel-title{font-size:21px;font-weight:800;margin-top:12px}.sell{color:#bc542f}.buy{color:#315f91}div[data-testid="stDataFrame"]{background:white;border-radius:10px}hr{border-color:#d6d0c9}</style>''', unsafe_allow_html=True)

REQUIRED = ['Ticker', 'Company', 'Sector', 'fiscal_end', 'STA', 'SNOA', 'B/M',
            'GPA', 'Altman Z', 'Accruals', 'Value Score', 'Quality Score',
            'Risk Score', 'Overall Score', 'Rank']
NUMERIC = ['STA', 'SNOA', 'B/M', 'GPA', 'Altman Z', 'Accruals',
           'Value Score', 'Quality Score', 'Risk Score', 'Overall Score', 'Rank']


def normalize_columns(data):
    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()
    aliases = {'ticker': 'Ticker', 'symbol': 'Ticker',
               'teaching_sector': 'Sector', 'sector': 'Sector',
               'company': 'Company', 'company_name': 'Company',
               'fiscal_end': 'fiscal_end'}
    data = data.rename(columns={c: aliases.get(c.lower(), c)
                                for c in data.columns})
    if 'Ticker' not in data.columns:
        raise ValueError('CSV must contain a Ticker (or ticker) column.')
    data['Ticker'] = data['Ticker'].astype(str).str.strip().str.upper()
    data = data[data['Ticker'].ne('') & data['Ticker'].ne('NAN')]
    return data.drop_duplicates('Ticker')


@st.cache_data(show_spinner=False)
def load_data():
    universe_path = Path(__file__).resolve().parent / 'universe.csv'
    if not universe_path.exists():
        return None
    base = normalize_columns(pd.read_csv(universe_path))
    for col in REQUIRED:
        if col not in base.columns:
            base[col] = np.nan
    base['Company'] = base['Company'].fillna(base['Ticker']).astype(str)
    base['Sector'] = base['Sector'].fillna('Unclassified').astype(str)
    base['fiscal_end'] = pd.to_datetime(base['fiscal_end'], errors='coerce')
    for col in NUMERIC:
        base[col] = pd.to_numeric(base[col], errors='coerce')
    if base['Accruals'].isna().all():
        base['Accruals'] = base[['STA', 'SNOA']].mean(axis=1)
    # Compute scores from the financial ratios if scores are not already present.
    def score(series, ascending=True):
        series = pd.to_numeric(series, errors='coerce').replace([np.inf,-np.inf],np.nan)
        if series.dropna().empty:
            return pd.Series(np.nan,index=series.index)
        if series.dropna().nunique() == 1:
            return pd.Series(np.where(series.notna(),50.0,np.nan),index=series.index)
        return series.rank(pct=True,ascending=ascending)*100
    if base['Value Score'].isna().all():
        base['Value Score'] = score(base['B/M'])
    if base['Quality Score'].isna().all():
        base['Quality Score'] = score(base['GPA'])
    if base['Risk Score'].isna().all():
        base['Risk Score'] = score(base['Altman Z'])
    if base['Overall Score'].isna().all():
        accrual = pd.concat([score(base['STA'],False),score(base['SNOA'],False)],axis=1).mean(axis=1)
        base['Overall Score'] = pd.concat([base['Value Score'],base['Quality Score'],base['Risk Score'],accrual],axis=1).mean(axis=1)
    if base['Rank'].isna().all():
        base['Rank'] = base['Overall Score'].rank(ascending=False,method='min')
    return base.reset_index(drop=True)

try:
    df = load_data()
except Exception as exc:
    st.error(f'Unable to read universe.csv: {exc}')
    st.stop()
if df is None:
    st.error('Upload **universe.csv** beside **app.py** on GitHub.')
    st.stop()
if df[['STA','SNOA','B/M','GPA','Altman Z']].notna().sum().sum() == 0:
    st.error('universe.csv has no financial ratios. A ticker-and-sector-only CSV cannot produce rankings without a data source. Use the financial rankings CSV renamed to universe.csv.')
    st.stop()

st.sidebar.markdown('## ⚙️ Screening controls')
st.sidebar.caption('Filters update the full ranking and all charts.')
all_sectors = sorted(df['Sector'].unique().tolist())
selected_sectors = st.sidebar.multiselect('Sectors to include', all_sectors, default=all_sectors)
min_coverage = st.sidebar.slider('Minimum ratio coverage (%)', 0, 100, 0, 20)
min_score = st.sidebar.slider('Minimum overall score', 0, 100, 0, 5)
top_n = st.sidebar.slider('Companies in ranking chart', 5, 40, 15, 5)
st.sidebar.divider()
st.sidebar.caption('Data source: universe.csv (saved snapshot, not a live feed).')

st.markdown('<div class="eyebrow">FINANCIAL ANALYSIS · INDUSTRIAL MANUFACTURING</div>', unsafe_allow_html=True)
st.title('Welcome to ManuValue!')
st.markdown('''ManuValue is an interactive stock screening tool that helps users explore industrial manufacturing companies and compare their financial strength. It analyzes **accruals (STA and SNOA)**, **book-to-market value**, **gross profitability**, and **Altman Z** to organize companies into financial rankings.

**How to use ManuValue:** Start with the sidebar to select manufacturing sectors, set a minimum financial-data coverage level, and adjust the minimum overall score. Explore the **all-company rankings table**, then scroll down to interact with full-width financial charts and investigate individual stocks. The methodology at the bottom explains how the financial ratios and scores were calculated.
''')
st.caption('Source: universe.csv · Saved financial statement snapshot, not live market data')

if not selected_sectors:
    st.info('Select at least one sector in the sidebar.')
    st.stop()

scope = df[df['Sector'].isin(selected_sectors)].copy()
metric_cols = ['STA', 'SNOA', 'B/M', 'GPA', 'Altman Z']
scope['Metric Coverage %'] = scope[metric_cols].notna().sum(axis=1) / len(metric_cols) * 100
scope = scope[(scope['Metric Coverage %'] >= min_coverage) &
              (scope['Overall Score'].isna() | (scope['Overall Score'] >= min_score))].copy()
if scope.empty:
    st.warning('No companies match these filters. Try lowering the minimum score or coverage.')
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric('Companies displayed', f'{len(scope):,}')
c2.metric('Companies in CSV', f'{len(df):,}')
c3.metric('With overall scores', f'{scope["Overall Score"].notna().sum():,}')

# FIRST: the entire stock ranking table.
st.divider()
st.subheader('All company stocks · financial rankings')
st.caption('Sorted by overall score. Scroll within the table or download the filtered data.')
columns = ['Ticker', 'Company', 'Sector', 'fiscal_end', 'STA', 'SNOA', 'B/M', 'GPA',
           'Altman Z', 'Accruals', 'Value Score', 'Quality Score', 'Risk Score',
           'Overall Score', 'Rank', 'Metric Coverage %']
if 'Score Coverage %' in scope.columns:
    columns.append('Score Coverage %')
if 'Data Status' in scope.columns:
    columns.append('Data Status')
full = scope[columns].sort_values('Overall Score', ascending=False, na_position='last').copy()
full['fiscal_end'] = full['fiscal_end'].dt.strftime('%Y-%m-%d')
st.dataframe(full, hide_index=True, use_container_width=True, height=510,
             column_config={name: st.column_config.NumberColumn(format='%.3f')
                            for name in ['STA', 'SNOA', 'B/M', 'GPA', 'Altman Z', 'Accruals']})
st.download_button('Download filtered company rankings (CSV)',
                   data=full.to_csv(index=False).encode('utf-8'),
                   file_name='manufacturing_screening_results.csv', mime='text/csv')

# SECOND: financial charts only (no BUY or SELL candidate charts/tables).
st.divider()
st.subheader('Financial charts')
st.markdown('### Overall stock rankings')
ranked = scope.dropna(subset=['Overall Score']).nlargest(top_n, 'Overall Score')
if not ranked.empty:
    fig = px.bar(ranked.sort_values('Overall Score'), x='Overall Score', y='Ticker',
                 orientation='h', hover_data=['Company', 'Sector'],
                 color='Overall Score', color_continuous_scale='Blues')
    fig.update_layout(height=max(520, 33 * len(ranked)), coloraxis_showscale=False,
                      margin=dict(l=20, r=20, t=25, b=25),
                      xaxis_title='Overall score (0–100)', yaxis_title='Company ticker')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No overall scores available for this selection.')

st.markdown('### Book-to-market vs. gross profitability')
st.caption('Hover over each point to identify the company. Use the legend to explore individual sectors.')
scatter = scope.dropna(subset=['B/M', 'GPA'])
if not scatter.empty:
    fig = px.scatter(scatter, x='B/M', y='GPA', color='Sector',
                     hover_name='Ticker', hover_data=['Company'],
                     custom_data=['Overall Score'])
    fig.update_traces(marker=dict(size=11, opacity=0.82))
    fig.update_layout(height=610, margin=dict(l=20, r=20, t=25, b=25),
                      legend_title_text='Manufacturing sector')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('Not enough B/M and GPA data for this chart.')

st.divider()
st.subheader('Individual company analysis')
st.markdown('**Explore one company**')
selected_ticker = st.selectbox('Select a ticker', sorted(scope['Ticker'].tolist()))
company = scope.loc[scope['Ticker'] == selected_ticker].iloc[0]
st.caption(f"{company['Company']} · {company['Sector']} · Fiscal end: " +
           (str(company['fiscal_end'].date()) if pd.notna(company['fiscal_end']) else 'Unavailable'))
metrics = st.columns(5)
for panel, name in zip(metrics, ['STA', 'SNOA', 'B/M', 'GPA', 'Altman Z']):
    value = company[name]
    panel.metric(name, f'{value:.3f}' if pd.notna(value) else 'N/A')
st.markdown('### Company category scores')
names = ['Value Score', 'Quality Score', 'Risk Score']
vals = pd.to_numeric(company[names], errors='coerce').dropna()
if not vals.empty:
    fig = px.bar(x=vals.index.tolist(), y=vals.values,
                 labels={'x': 'Category', 'y': 'Score / 100'},
                 color=vals.index.tolist(), color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=490, margin=dict(l=20, r=20, t=25, b=25),
                      showlegend=False, yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No category scores available.')

st.markdown('### Company accrual ratios')
names = ['STA', 'SNOA', 'Accruals']
vals = pd.to_numeric(company[names], errors='coerce').dropna()
if not vals.empty:
    fig = px.bar(x=vals.index.tolist(), y=vals.values,
                 labels={'x': 'Ratio', 'y': 'Value'},
                 color=vals.index.tolist(), color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=490, margin=dict(l=20, r=20, t=25, b=25),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No accrual ratios available.')

# THIRD: methodology last.
st.divider()
st.subheader('Methodology')
st.markdown('''**Financial ratios:** STA = (net income − operating cash flow) / total assets (a cash-flow accrual proxy); SNOA = simplified net operating assets / total assets; B/M = book equity / market capitalization; GPA = gross profit / total assets. **Accruals** is the custom average of STA and SNOA, not a standard accounting ratio.

**Scores and ranking:** Value Score ranks B/M (higher preferred); Quality Score ranks GPA (higher preferred); Risk Score ranks Altman Z (higher preferred). The overall score in the saved CSV combines value, quality, accrual scoring (lower STA and SNOA preferred), and risk, with equal weights across available categories. Rankings and scores are from the original dataset; filtering does not recalculate their percentiles.

**Altman Z:** The original Colab dataset used course-supplied coefficients. Its values should not be compared with standard bankruptcy cutoffs without verifying the scaling and model variant.

**Limitations:** These are historical financial statement values saved in `universe.csv`. Market capitalization and annual accounting dates may differ. This educational screener does not provide live market data or investment advice.''')
st.caption('Source: universe.csv · Saved classroom financial data · Educational use only')
