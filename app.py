import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title='Industrial Manufacturing | Stock Screener', page_icon='🏭', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:#f5f3ef;color:#17243b}h1,h2,h3{color:#17243b!important}
section[data-testid="stSidebar"]{background:#19243b}section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p{color:#f5f7fc!important}
section[data-testid="stSidebar"] [data-baseweb="select"] *{color:#18243e!important}
div[data-testid="stMetric"]{background:white;border:1px solid #e1ddd7;border-radius:12px;padding:12px 16px}
div[data-testid="stDataFrame"]{background:white;border-radius:12px}.eyebrow{font-size:13px;font-weight:800;letter-spacing:2px;color:#ad512d}.intro{font-size:17px;line-height:1.7;max-width:1100px;color:#34435b}.sidebar-section-heading{color:#ffffff!important;font-size:1.15rem;font-weight:750;line-height:1.5;margin:1.3rem 0 .45rem;padding:.5rem .75rem;background:#344666;border-left:4px solid #f5a65b;border-radius:6px}</style>''', unsafe_allow_html=True)

RATIOS = ['STA','SNOA','B/M','GPA','Altman Z','Accruals']
CATEGORIES = ['Value','Quality','Accrual Quality','Financial Resilience']

@st.cache_data(show_spinner=False)
def load_data():
    path = Path(__file__).resolve().parent / 'universe.csv'
    if not path.exists():
        raise FileNotFoundError('Place universe.csv beside app.py in your GitHub repository.')
    d = pd.read_csv(path)
    d.columns = d.columns.astype(str).str.strip()
    aliases = {'ticker':'Ticker','symbol':'Ticker','company':'Company','sector':'Sector','teaching_sector':'Sector','fiscal_end':'fiscal_end','fiscal_end_date':'fiscal_end'}
    d = d.rename(columns={c:aliases.get(c.lower(),c) for c in d.columns})
    if 'Ticker' not in d:
        raise ValueError('universe.csv needs a Ticker column.')
    d['Ticker'] = d['Ticker'].astype(str).str.strip().str.upper()
    d = d[d['Ticker'].ne('') & d['Ticker'].ne('NAN')].drop_duplicates('Ticker').copy()
    for col in ['Company','Sector','fiscal_end']+RATIOS+['Value Score','Quality Score','Risk Score','Overall Score','Rank']:
        if col not in d:
            d[col] = np.nan
    d['Company'] = d['Company'].fillna(d['Ticker'])
    d['Sector'] = d['Sector'].fillna('Unclassified')
    d['fiscal_end'] = pd.to_datetime(d['fiscal_end'],errors='coerce')
    for col in RATIOS+['Value Score','Quality Score','Risk Score','Overall Score','Rank']:
        d[col] = pd.to_numeric(d[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    d['Accruals'] = d['Accruals'].fillna(d[['STA','SNOA']].mean(axis=1))
    return d.reset_index(drop=True)

try:
    data = load_data()
except Exception as e:
    st.error(str(e)); st.stop()
if data[['STA','SNOA','B/M','GPA','Altman Z']].notna().sum().sum() == 0:
    st.error('universe.csv contains only company names/tickers, not financial ratios. Upload the financial-data version of universe.csv.'); st.stop()

def percentile(series, higher=True):
    v = pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan)
    if not higher: v = -v
    if v.dropna().empty: return pd.Series(np.nan,index=v.index)
    if v.dropna().nunique() == 1: return pd.Series(np.where(v.notna(),50.,np.nan),index=v.index)
    return v.rank(pct=True,method='average')*100

# Score inputs are calculated from the complete universe and stay comparable when filters change.
data['_Value'] = percentile(data['B/M'])
data['_Quality'] = percentile(data['GPA'])
data['_Accrual'] = pd.concat([percentile(data['STA'],False),percentile(data['SNOA'],False)],axis=1).mean(axis=1)
data['_Resilience'] = percentile(data['Altman Z'])
data['Coverage %'] = data[['STA','SNOA','B/M','GPA','Altman Z']].notna().sum(axis=1)*20

st.sidebar.title('Explore & filter')
st.sidebar.caption('Every change updates the table and financial charts.')
sectors = sorted(data['Sector'].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect('Sectors to include',sectors,default=sectors)
st.sidebar.markdown('<div class="sidebar-section-heading">Build your own ranking</div>', unsafe_allow_html=True)
st.sidebar.caption('Adjust the importance of each category. Weights are normalized automatically.')
w_value=st.sidebar.slider('Value weight',0,100,25,5)
w_quality=st.sidebar.slider('Profitability weight',0,100,25,5)
w_accrual=st.sidebar.slider('Accrual quality weight',0,100,25,5)
w_risk=st.sidebar.slider('Financial resilience weight',0,100,25,5)
weights=np.array([w_value,w_quality,w_accrual,w_risk],dtype=float)
if weights.sum()==0:
    st.sidebar.warning('All weights are zero. Using equal weights until one is increased.')
    weights=np.array([25.,25.,25.,25.])
st.sidebar.caption('Effective weights: '+', '.join(f'{name} {w/weights.sum():.0%}' for name,w in zip(CATEGORIES,weights)))
st.sidebar.markdown('<div class="sidebar-section-heading">BUY / SELL screening</div>', unsafe_allow_html=True)
st.sidebar.caption('SELL uses accrual quality. BUY uses value and profitability — different financial tests.')
sell_pct = st.sidebar.slider('SELL: high accrual percentile', 50, 95, 80, 5, help='Flag unusually high STA and/or SNOA values compared with the full company universe.')
sell_signals = st.sidebar.select_slider('SELL: accrual measures required', options=[1, 2], value=1, help='Require either STA or SNOA to be high, or require both.')
buy_pct = st.sidebar.slider('BUY: minimum value & profitability percentile', 50, 95, 70, 5, help='A BUY candidate must clear this threshold for BOTH book-to-market and gross profitability.')
chart_group = st.sidebar.radio('Stocks shown in financial charts', ['All Stocks', 'BUY Candidates', 'SELL Candidates', 'Neutral'], index=0)
st.sidebar.markdown('<div class="sidebar-section-heading">Narrow your results</div>', unsafe_allow_html=True)
min_coverage=st.sidebar.slider('Minimum ratio coverage (%)',0,100,0,20)
min_score=st.sidebar.slider('Minimum custom score',0,100,0,5)
score_limit=st.sidebar.slider('Companies in ranking chart',5,50,20,5)
st.sidebar.markdown('<div class="sidebar-section-heading">Chart playground</div>', unsafe_allow_html=True)
chart_metric=st.sidebar.selectbox('Ranking chart metric',['Custom Score','B/M','GPA','STA','SNOA','Accruals','Altman Z'])
chart_style=st.sidebar.radio('Ranking chart style',['Horizontal bars','Vertical bars'],horizontal=False)
scatter_x=st.sidebar.selectbox('Scatter X axis',RATIOS+['Custom Score'],index=2)
scatter_y=st.sidebar.selectbox('Scatter Y axis',RATIOS+['Custom Score'],index=3)
color_choice=st.sidebar.selectbox('Scatter colour by',['Sector','Custom Score','Coverage %'])
show_labels=st.sidebar.checkbox('Show ticker labels on scatter',value=False)
st.sidebar.caption('Source: universe.csv · Saved financial snapshot, not live market data')

# Reweight all available score components per company, without treating missing as zero.
score_values=data[['_Value','_Quality','_Accrual','_Resilience']].to_numpy(dtype=float)
valid=np.isfinite(score_values)
weighted=np.where(valid,score_values,0)*weights
weight_available=(valid*weights).sum(axis=1)
data['Custom Score']=np.divide(weighted.sum(axis=1),weight_available,out=np.full(len(data),np.nan),where=weight_available>0)
data['Custom Rank']=data['Custom Score'].rank(ascending=False,method='min').astype('Int64')
data['Score Coverage %']=(valid* (weights>0)).sum(axis=1)/max((weights>0).sum(),1)*100

# Distinct investment screens (independent of the customizable overall ranking).
# SELL: high raw STA / SNOA accrual ratios are adverse; either or both may be required.
# BUY: BOTH value and gross profitability must be high. This is not the inverse of SELL.
# Missing metrics never count as passing a test.
data['_STA_High'] = percentile(data['STA']).ge(sell_pct).fillna(False)
data['_SNOA_High'] = percentile(data['SNOA']).ge(sell_pct).fillna(False)
data['_SELL_Signals'] = data[['_STA_High', '_SNOA_High']].sum(axis=1)
data['_SELL'] = data['_SELL_Signals'].ge(sell_signals)
data['_BUY_Value'] = data['_Value'].ge(buy_pct).fillna(False)
data['_BUY_Profit'] = data['_Quality'].ge(buy_pct).fillna(False)
data['_BUY'] = data['_BUY_Value'] & data['_BUY_Profit'] & ~data['_SELL']
data['Screening Signal'] = np.select(
    [data['_SELL'], data['_BUY']], ['SELL Candidate', 'BUY Candidate'], default='Neutral'
)

st.markdown('<div class="eyebrow">FINANCIAL DISCOVERY · INDUSTRIAL MANUFACTURING</div>',unsafe_allow_html=True)
st.title('Industrial Manufacturing')
st.markdown('''<div class="intro">Explore industrial manufacturing stocks through an interactive financial lens. Compare company valuations, profitability, and accruals, then build your own ranking by deciding which factors matter most. <strong>The results change as you experiment</strong>—use the controls on the left to test ideas, discover unusual companies and investigate what drives their scores.</div>''',unsafe_allow_html=True)
st.caption('Educational stock screening based on the historical financial figures in universe.csv. Scores are relative to the saved universe, not forecasts or investment advice.')

if not selected_sectors:
    st.info('Select at least one sector to display companies.');st.stop()
scope=data[data['Sector'].isin(selected_sectors)].copy()
scope=scope[(scope['Coverage %']>=min_coverage)&(scope['Custom Score'].isna() | (scope['Custom Score']>=min_score))].copy()
if scope.empty:
    st.warning('No companies match the current filters. Try a lower minimum score or more sectors.');st.stop()

m1,m2,m3,m4=st.columns(4)
m1.metric('Companies on screen',len(scope))
m2.metric('BUY candidates',int((scope['Screening Signal']=='BUY Candidate').sum()))
m3.metric('SELL candidates',int((scope['Screening Signal']=='SELL Candidate').sum()))
m4.metric('Neutral companies',int((scope['Screening Signal']=='Neutral').sum()))

st.divider()
st.header('All company stocks')
st.write('Sort any column, search the table or change the sidebar to instantly update the universe. **Custom Score** and **Custom Rank** respond to your chosen factor weights.')
search=st.text_input('Find a company or ticker',placeholder='e.g. CAT, Deere, Honeywell')
if search.strip():
    scope=scope[scope['Ticker'].str.contains(search.strip(),case=False,regex=False,na=False)|scope['Company'].astype(str).str.contains(search.strip(),case=False,regex=False,na=False)].copy()
if scope.empty:
    st.info('No matching tickers or companies.');st.stop()
cols=['Ticker','Company','Sector','fiscal_end','STA','SNOA','B/M','GPA','Altman Z','Accruals','Screening Signal','Value Score','Quality Score','Risk Score','Overall Score','Custom Score','Custom Rank','Rank','Coverage %','Score Coverage %']
cols=[c for c in cols if c in scope.columns]
table=scope[cols].sort_values('Custom Score',ascending=False,na_position='last').copy()
table['fiscal_end']=table['fiscal_end'].dt.strftime('%Y-%m-%d')
st.dataframe(table,use_container_width=True,hide_index=True,height=510)
st.download_button('Download filtered table (CSV)',table.to_csv(index=False).encode('utf-8'),'industrial_equity_screen.csv','text/csv')

st.divider()
st.header('Financial chart playground')
st.caption('Charts use the selected sectors, score filters and BUY/SELL category. The all-company table above stays visible.')
chart_scope = scope.copy()
if chart_group != 'All Stocks':
    target = {'BUY Candidates':'BUY Candidate', 'SELL Candidates':'SELL Candidate', 'Neutral':'Neutral'}[chart_group]
    chart_scope = chart_scope[chart_scope['Screening Signal']==target].copy()
st.caption(f'Chart view: {chart_group} · {len(chart_scope)} companies')
if chart_scope.empty:
    st.info('No stocks match this chart category with the current screening settings. Adjust the BUY/SELL thresholds or choose All Stocks.')

st.subheader(f'Company comparison · {chart_metric}')
ascending=chart_metric in ['STA','SNOA','Accruals']
ranked=chart_scope.dropna(subset=[chart_metric]).sort_values(chart_metric,ascending=ascending).head(score_limit)
if not ranked.empty:
    if chart_style=='Horizontal bars':
        fig=px.bar(ranked.sort_values(chart_metric,ascending=not ascending),x=chart_metric,y='Ticker',orientation='h',color=chart_metric,color_continuous_scale='Blues',hover_data=['Company','Sector','Custom Score'])
    else:
        fig=px.bar(ranked,x='Ticker',y=chart_metric,color=chart_metric,color_continuous_scale='Blues',hover_data=['Company','Sector','Custom Score'])
    fig.update_layout(height=max(420,min(650,25*len(ranked))),margin=dict(l=15,r=15,t=30,b=35),coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
else:st.info('No valid data for the selected metric.')

st.subheader(f'Explore relationships · {scatter_x} vs {scatter_y}')
plot=chart_scope.dropna(subset=[scatter_x,scatter_y]).copy()
if not plot.empty:
    fig=px.scatter(plot,x=scatter_x,y=scatter_y,color=color_choice,hover_name='Ticker',hover_data=['Company','Sector','Custom Score'],text='Ticker' if show_labels else None,opacity=.8,size_max=18)
    fig.update_traces(marker=dict(size=12),textposition='top center')
    fig.update_layout(height=490,margin=dict(l=15,r=15,t=30,b=30))
    st.plotly_chart(fig,use_container_width=True)
else:st.info('No companies have both selected financial measures.')

st.subheader('How do manufacturing groups compare?')
sector_metric=st.selectbox('Financial measure for sector comparison',['Custom Score','B/M','GPA','STA','SNOA','Accruals','Altman Z'],key='sector_metric')
sector_data=chart_scope.dropna(subset=[sector_metric])
if not sector_data.empty:
    fig=px.box(sector_data,x='Sector',y=sector_metric,color='Sector',points='all',hover_data=['Ticker','Company'])
    fig.update_layout(height=490,showlegend=False,margin=dict(l=15,r=15,t=30,b=95),xaxis_tickangle=-18)
    st.plotly_chart(fig,use_container_width=True)
else:st.info('Not enough data for sector comparisons.')

st.divider()
st.header('Company deep dive')
selected=st.selectbox('Choose a company',sorted(scope['Ticker'].tolist()))
company=scope.loc[scope['Ticker']==selected].iloc[0]
st.subheader(f'{company["Company"]} ({selected})')
st.caption(f'{company["Sector"]} · Fiscal year end: '+(company['fiscal_end'].strftime('%Y-%m-%d') if pd.notna(company['fiscal_end']) else 'Not available'))

signal = company['Screening Signal']
if signal == 'SELL Candidate':
    reasons = []
    if bool(company['_STA_High']):
        reasons.append(f"STA = {company['STA']:.4f}, in the high-accrual group (at or above the {sell_pct}th percentile)")
    if bool(company['_SNOA_High']):
        reasons.append(f"SNOA = {company['SNOA']:.4f}, in the high-accrual group (at or above the {sell_pct}th percentile)")
    st.error('**SELL candidate — accrual-based earnings-quality screen.** ' + '; '.join(reasons) + '. Higher accruals can warrant closer investigation, but do not prove overvaluation or a future price decline.')
elif signal == 'BUY Candidate':
    st.success(f"**BUY candidate — value and profitability screen.** Book-to-market = {company['B/M']:.4f} ({company['_Value']:.1f}th percentile) and GPA = {company['GPA']:.4f} ({company['_Quality']:.1f}th percentile). Both meet the {buy_pct}th-percentile requirement, and the company does not meet the SELL accrual screen. This is a screening hypothesis, not proof of undervaluation.")
else:
    why = []
    if pd.isna(company['B/M']) or pd.isna(company['GPA']):
        why.append('book-to-market or gross profitability data is missing')
    elif not (bool(company['_BUY_Value']) and bool(company['_BUY_Profit'])):
        why.append(f"B/M and GPA do not both reach the {buy_pct}th-percentile BUY threshold")
    if not bool(company['_SELL']):
        why.append(f"fewer than {sell_signals} accrual measures meet the {sell_pct}th-percentile SELL threshold")
    st.info('**Neutral — neither screening rule is met.** ' + '; '.join(why) + '. Adjust the sidebar thresholds to test different screening decisions.')

c1,c2,c3,c4=st.columns(4)
c1.metric('Custom Score',f'{company["Custom Score"]:.1f}' if pd.notna(company['Custom Score']) else 'N/A')
c2.metric('Custom Rank',str(company['Custom Rank']) if pd.notna(company['Custom Rank']) else 'N/A')
c3.metric('Overall Score',f'{company["Overall Score"]:.1f}' if pd.notna(company['Overall Score']) else 'N/A')
c4.metric('Ratio Coverage',f'{company["Coverage %"]:.0f}%')
st.subheader('Financial ratio profile')
ratios=pd.DataFrame({'Metric':RATIOS,'Value':[company[m] for m in RATIOS]}).dropna()
if not ratios.empty:
    fig=px.bar(ratios,x='Metric',y='Value',color='Metric',text_auto='.3f')
    fig.update_layout(height=410,showlegend=False,margin=dict(l=15,r=15,t=30,b=30))
    st.plotly_chart(fig,use_container_width=True)
st.subheader('What drives this company’s ranking?')
parts=pd.DataFrame({'Category':CATEGORIES,'Percentile Score':[company['_Value'],company['_Quality'],company['_Accrual'],company['_Resilience']],'Weight %':weights/weights.sum()*100}).dropna(subset=['Percentile Score'])
if not parts.empty:
    fig=px.bar(parts,x='Category',y='Percentile Score',color='Weight %',text='Weight %',color_continuous_scale='Blues')
    fig.update_traces(texttemplate='%{text:.0f}% weight',textposition='outside')
    fig.update_layout(height=430,yaxis_range=[0,112],margin=dict(l=15,r=15,t=30,b=30))
    st.plotly_chart(fig,use_container_width=True)

st.divider()
st.header('Methodology & interpretation')
st.markdown(f'''**BUY / SELL methodology.** **SELL** is based on unusually high STA and/or SNOA accrual ratios, requiring at least **{sell_signals}** measure(s) at or above the **{sell_pct}th percentile** of the complete CSV universe. **BUY** requires **both** B/M and GPA at or above the **{buy_pct}th percentile**, with no qualifying SELL accrual signal. These are **different** analytical screens: earnings-quality concerns for SELL, value plus gross profitability for BUY. Stocks that qualify for neither are **Neutral**. A company satisfying both screens is classified SELL to avoid contradictory labels. Chart category filters change chart membership, while the full company table remains visible. The customizable ranking weights affect Custom Score and Custom Rank, not the separate BUY/SELL criteria.

**Financial measures.** STA is the net-income-minus-operating-cash-flow accrual proxy divided by total assets (not a pure short-term accrual decomposition). SNOA is a simplified net operating assets ratio. B/M = book equity / market capitalization. GPA = gross profit / total assets. Accruals is the custom average of STA and SNOA; it is not a standardized accounting ratio.

**Ranking model.** Each metric is converted to a percentile relative to the complete CSV universe. Higher B/M, GPA and Altman Z are preferred; lower STA and SNOA are preferred. The accrual-quality component averages the STA and SNOA percentiles. Your sidebar weights combine the four components into **Custom Score**, normalized across the available nonmissing components. **Custom Rank** changes when you change weights, while the original **Overall Score** and **Rank** are preserved from your CSV.

**Limitations.** These are historical annual figures saved in `universe.csv`, not automatically refreshed market data. The source dataset uses a simplified SNOA calculation and a course-specific Altman Z formula; standard bankruptcy thresholds may not apply. A high screening score is not automatically a BUY recommendation. Some companies have incomplete metrics; use coverage information when comparing rankings.''')
