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
    ranking_path = Path(__file__).resolve().parent / 'manufacturing_175_rankings.csv'

    if universe_path.exists():
        base = normalize_columns(pd.read_csv(universe_path))
        source = 'universe.csv'
        if ranking_path.exists():
            rankings = normalize_columns(pd.read_csv(ranking_path))
            # Universe defines membership. Ranking CSV supplies missing financial columns.
            merge_cols = ['Ticker'] + [c for c in rankings.columns
                                       if c != 'Ticker' and c not in base.columns]
            base = base.merge(rankings[merge_cols], on='Ticker', how='left')
            # If universe contains its own financial columns, prefer those values.
            source += ' + manufacturing_175_rankings.csv'
    elif ranking_path.exists():
        base = normalize_columns(pd.read_csv(ranking_path))
        source = 'manufacturing_175_rankings.csv'
    else:
        return None, None

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
    # Scores cannot be reconstructed from ticker-only files: show missing honestly.
    return base.reset_index(drop=True), source


try:
    df, data_source = load_data()
except Exception as exc:
    st.error(f'Unable to read your CSV file: {exc}')
    st.stop()
if df is None:
    st.error('Upload **universe.csv** beside **app.py** on GitHub. '
             'Optionally upload **manufacturing_175_rankings.csv** for ratios and scores.')
    st.stop()

st.sidebar.markdown('## ⚙️ Screening controls')
st.sidebar.caption('Every setting updates the dashboard.')
all_sectors=sorted(df['Sector'].unique().tolist())
selected_sectors=st.sidebar.multiselect('Sectors to include',all_sectors,default=all_sectors)
accrual_percentile=st.sidebar.slider('High accruals flag: percentile',50,95,75,5,help='Flag companies whose combined accruals are at or above this percentile among the selected companies.')
z_cutoff=st.sidebar.number_input('Low Altman Z flag: below',min_value=0.0,max_value=10.0,value=1.8,step=0.1,help='Custom screening cutoff for the course-scaled Z measure; NOT a validated bankruptcy threshold.')
flags_required=st.sidebar.slider('Red flags required for SELL',1,4,2)
buy_percentile=st.sidebar.slider('BUY minimum overall-score percentile',50,95,75,5)
min_coverage=st.sidebar.slider('Minimum metric coverage (%)',0,100,60,20)
max_rows=st.sidebar.slider('Rows in each candidate table',5,50,15,5)
st.sidebar.divider()
st.sidebar.caption(f'Data loaded from: {data_source}. CSV values are a saved snapshot, not a live Yahoo Finance feed.')

st.markdown('<div class="eyebrow">FINANCIAL ANALYSIS · INDUSTRIAL MANUFACTURING</div>',unsafe_allow_html=True)
st.title('Manufacturing Stock Screener')
st.markdown('<p class="subtle">Filters on the left drive BUY and SELL screening, company comparisons, and financial analysis on the right.</p>',unsafe_allow_html=True)

if not selected_sectors:
    st.info('Select at least one sector in the sidebar.')
    st.stop()

scope=df[df['Sector'].isin(selected_sectors)].copy()
metric_cols=['STA','SNOA','B/M','GPA','Altman Z']
scope['Metric Coverage %']=scope[metric_cols].notna().sum(axis=1)/len(metric_cols)*100
scope=scope[scope['Metric Coverage %']>=min_coverage].copy()
if scope.empty:
    st.warning('No companies match your sector and data-coverage filters. Try reducing minimum coverage.')
    st.stop()

# Recalculate thresholds within the selected, data-qualified universe.
accrual_threshold=scope['Accruals'].quantile(accrual_percentile/100)
buy_threshold=scope['Overall Score'].quantile(buy_percentile/100)
# Relative risk and value flags are calculated only where data exists.
low_quality_threshold=scope['GPA'].quantile(.25)
low_value_threshold=scope['B/M'].quantile(.25)

scope['High accruals']=scope['Accruals'].notna() & (scope['Accruals']>=accrual_threshold)
scope['Low Altman Z']=scope['Altman Z'].notna() & (scope['Altman Z']<z_cutoff)
scope['Low profitability']=scope['GPA'].notna() & (scope['GPA']<=low_quality_threshold)
scope['Low book-to-market']=scope['B/M'].notna() & (scope['B/M']<=low_value_threshold)
flag_cols=['High accruals','Low Altman Z','Low profitability','Low book-to-market']
scope['Red Flags']=scope[flag_cols].sum(axis=1)
scope['Flag Details']=scope.apply(lambda r:', '.join(label for col,label in zip(flag_cols,['High accruals','Low Z','Low GPA','Low B/M']) if r[col]) or 'None',axis=1)

# BUY and SELL use separate rules, and exclude contradictory BUY recommendations.
sells=scope[scope['Red Flags']>=flags_required].sort_values(['Red Flags','Overall Score'],ascending=[False,True]).copy()
buys=scope[(scope['Overall Score']>=buy_threshold)&(scope['Red Flags']==0)&scope['Overall Score'].notna()].sort_values('Overall Score',ascending=False).copy()

c1,c2,c3,c4=st.columns(4)
c1.metric('Companies screened',f'{len(scope):,}')
c2.metric('SELL candidates',f'{len(sells):,}')
c3.metric('BUY candidates',f'{len(buys):,}')
c4.metric('Universe in file',f'{len(df):,}')
if df['Overall Score'].notna().sum() == 0:
    st.warning('No overall scores were found. If universe.csv contains only ticker/sector, upload manufacturing_175_rankings.csv as well to enable BUY/SELL rankings.')
st.caption(f'{len(df)} tickers in the selected universe; {df["Overall Score"].notna().sum()} have overall scores.')

st.divider()
left,right=st.columns(2,gap='medium')
with left:
    st.markdown('<div class="panel-title sell">SELL candidates</div>',unsafe_allow_html=True)
    st.caption('Companies with multiple financial red flags; highest flag count first.')
    sell_columns=['Ticker','Company','Sector','Red Flags','Flag Details','STA','SNOA','Accruals','GPA','Altman Z','Overall Score']
    st.dataframe(sells[sell_columns].head(max_rows),hide_index=True,use_container_width=True,height=360,column_config={'Overall Score':st.column_config.NumberColumn(format='%.1f'),'STA':st.column_config.NumberColumn(format='%.3f'),'SNOA':st.column_config.NumberColumn(format='%.3f'),'Accruals':st.column_config.NumberColumn(format='%.3f'),'GPA':st.column_config.NumberColumn(format='%.3f'),'Altman Z':st.column_config.NumberColumn(format='%.3f')})
    if sells.empty:st.info('No SELL candidates meet the current flag requirement.')
with right:
    st.markdown('<div class="panel-title buy">BUY candidates</div>',unsafe_allow_html=True)
    st.caption('High overall score and zero red flags; strongest scores first.')
    buy_columns=['Ticker','Company','Sector','B/M','GPA','Altman Z','Value Score','Quality Score','Risk Score','Overall Score']
    st.dataframe(buys[buy_columns].head(max_rows),hide_index=True,use_container_width=True,height=360,column_config={'Overall Score':st.column_config.NumberColumn(format='%.1f'),'B/M':st.column_config.NumberColumn(format='%.3f'),'GPA':st.column_config.NumberColumn(format='%.3f'),'Altman Z':st.column_config.NumberColumn(format='%.3f')})
    if buys.empty:st.info('No BUY candidates satisfy both the score threshold and zero-red-flags rule.')

st.divider()
st.subheader('Financial screening charts')
chart_left,chart_right=st.columns(2,gap='medium')
with chart_left:
    st.markdown('**Highest overall scores**')
    ranked = scope.dropna(subset=['Overall Score']).nlargest(15, 'Overall Score')
    if not ranked.empty:
        fig = px.bar(ranked.sort_values('Overall Score'), x='Overall Score', y='Ticker',
                     orientation='h', hover_data=['Company', 'Sector'],
                     color='Overall Score', color_continuous_scale='Blues')
        fig.update_layout(height=430, coloraxis_showscale=False,
                          margin=dict(l=5, r=5, t=10, b=5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No overall scores are available for these companies.')
with chart_right:
    st.markdown('**Value vs. profitability**')
    scatter = scope.dropna(subset=['B/M', 'GPA']).copy()
    if not scatter.empty:
        fig = px.scatter(scatter, x='B/M', y='GPA', color='Sector',
                         hover_name='Ticker', hover_data=['Company'],
                         title=None)
        fig.update_layout(height=430, margin=dict(l=5, r=5, t=10, b=5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Not enough B/M and GPA observations for this chart.')

st.divider()
st.subheader('Company drill-down')
company_options=sorted(scope['Ticker'].tolist())
selected_ticker=st.selectbox('Choose a company',company_options,index=0)
company=scope.loc[scope['Ticker']==selected_ticker].iloc[0]
st.markdown(f"**{company['Company']} ({selected_ticker})** · {company['Sector']}")
end=company['fiscal_end']
st.caption(f"Fiscal year end: {end.date() if pd.notna(end) else 'Unavailable'} · Screening red flags: {int(company['Red Flags'])} ({company['Flag Details']})")
metric_row=st.columns(5)
for box,(name,fmt) in zip(metric_row,[('STA','.3f'),('SNOA','.3f'),('B/M','.3f'),('GPA','.3f'),('Altman Z','.3f')]):
    value=company[name]
    box.metric(name,format(value,fmt) if pd.notna(value) else 'N/A')
chart1, chart2 = st.columns(2)
with chart1:
    st.markdown('**Company financial scores**')
    categories = ['Value Score', 'Quality Score', 'Risk Score']
    score_values = pd.to_numeric(company[categories], errors='coerce').dropna()
    if not score_values.empty:
        fig = px.bar(x=score_values.index, y=score_values.values, labels={'x':'Category','y':'Score / 100'})
        fig.update_layout(height=340, margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No financial category scores available.')
with chart2:
    st.markdown('**Accrual-related ratios**')
    accrual_values = pd.to_numeric(company[['STA', 'SNOA', 'Accruals']], errors='coerce').dropna()
    if not accrual_values.empty:
        fig = px.bar(x=accrual_values.index, y=accrual_values.values, labels={'x':'Metric','y':'Ratio'})
        fig.update_layout(height=340, margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No accrual-related ratios available.')

st.divider()
st.subheader('All companies · interactive ranking table')
all_cols=['Ticker','Company','Sector','fiscal_end','STA','SNOA','B/M','GPA','Altman Z','Accruals','Value Score','Quality Score','Risk Score','Overall Score','Rank','Red Flags','Flag Details']
full=scope[all_cols].sort_values('Overall Score',ascending=False).copy()
full['fiscal_end']=full['fiscal_end'].dt.strftime('%Y-%m-%d')
st.dataframe(full,hide_index=True,use_container_width=True,height=430)
st.download_button('Download filtered screening results (CSV)',data=full.to_csv(index=False).encode('utf-8'),file_name='manufacturing_screening_results.csv',mime='text/csv')

st.divider()
st.subheader('Conclusions and methodology')
st.markdown('''**BUY screen:** A company must rank above the chosen overall-score percentile and have **zero** of the four defined red flags. **SELL screen:** A company must have at least the selected number of red flags. Companies can appear in neither list; the screens are intentionally separate.

**Red flags:** (1) combined accruals at or above the selected percentile, (2) Altman Z below the chosen cutoff, (3) GPA in the bottom quartile of the selected universe, and (4) B/M in the bottom quartile. The accrual and quartile thresholds are recalculated whenever the sector or data-coverage filters change.

**Financial ratios:** STA = (net income − operating cash flow) / total assets; SNOA is a simplified net operating assets proxy / total assets; B/M = book equity / market capitalization; GPA = gross profit / total assets. **Accruals** in this dataset is the custom average of STA and SNOA, not a standard financial-statement ratio.

**Overall Score:** The uploaded score was calculated from relative percentile rankings of value (B/M), quality (GPA), accruals (STA and SNOA, with lower preferred), and risk (Altman Z). The four categories were equally weighted where data was available. This app preserves the uploaded scores; filtering changes the screen thresholds, not the underlying original ranking scores.

**Important limitations:** This is an educational screening model, not a trading recommendation. Annual accounting figures are historical; B/M and Altman Z may use market capitalization from a different date. The supplied Altman Z uses course-specific coefficients, so the cutoff here is a **user-adjustable heuristic**, not a validated distress or bankruptcy threshold. Some tickers may be missing or stale; no live-price or live-financial refresh occurs in this version.''')
st.caption(f'Data source: {data_source} · Saved classroom data, not live prices · Educational use only')
