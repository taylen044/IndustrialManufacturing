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
    st.error('universe.csv contains only company names/tickers, not financial ratios. Upload the financial data version of universe.csv.'); st.stop()

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
st.sidebar.caption('Adjust the screening sensitivity to explore potential opportunities and risks.')
sell_pct = st.sidebar.slider('SELL minimum financial weakness', 50, 95, 75, 5, help='Lower thresholds identify more potential SELL candidates. Screening combines multiple financial measures.')
buy_pct = st.sidebar.slider('BUY minimum financial strength', 50, 95, 75, 5, help='Lower thresholds identify more potential BUY candidates. Screening combines multiple financial measures.')
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

# Both screens consider all five underlying financial ratios, but use
# distinct analytical emphases rather than simply reversing one score.
# SELL emphasizes accrual risk and financial weakness; BUY emphasizes
# value and profitability supported by financial strength and earnings quality.
# A component's missing value is not silently treated as a passing signal.
def weighted_screen(components):
    frame = pd.DataFrame(components, index=data.index)
    return frame.mean(axis=1, skipna=False)

# High scores indicate stronger evidence for the specified screen.
data['_SellStrength'] = weighted_screen({
    'STA concern': percentile(data['STA']),
    'SNOA concern': percentile(data['SNOA']),
    'Low profitability': percentile(data['GPA'], False),
    'Low resilience': percentile(data['Altman Z'], False),
    'Low value': percentile(data['B/M'], False),
})
data['_BuyStrength'] = weighted_screen({
    'Attractive value': data['_Value'],
    'High profitability': data['_Quality'],
    'Financial resilience': data['_Resilience'],
    'Cash earnings quality': data['_Accrual'],
})

# The SELL screen uses all five raw ratios; the BUY screen uses B/M,
# GPA, Altman Z and both accrual components. Different weights make the
# screens independent investment theses, not simple opposites.
# Require above-median core evidence in addition to composite strength.
data['_SELL'] = (data['_SellStrength'] >= sell_pct) & (
    (percentile(data['STA']) >= 60) | (percentile(data['SNOA']) >= 60)
)
data['_BUY'] = (data['_BuyStrength'] >= buy_pct) & (
    (data['_Value'] >= 55) & (data['_Quality'] >= 55)
)
# Avoid conflicting recommendations; mixed evidence is Neutral.
data['Screening Signal'] = np.select(
    [data['_SELL'] & ~data['_BUY'], data['_BUY'] & ~data['_SELL']],
    ['SELL Candidate', 'BUY Candidate'], default='Neutral'
)

st.markdown('<div class="eyebrow">FINANCIAL DISCOVERY · INDUSTRIAL MANUFACTURING</div>',unsafe_allow_html=True)
st.title('Industrial Manufacturing')
st.markdown('''<div class="intro">Explore industrial manufacturing stocks through an interactive financial lens. Compare company valuations, profitability, and accruals, then build your own ranking by deciding which factors matter most. <strong>The results change as you experiment</strong>, use the controls on the left to test ideas, discover unusual companies and investigate what drives their scores.</div>''',unsafe_allow_html=True)
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
cols=['Ticker','Company','Sector','fiscal_end','STA','SNOA','B/M','GPA','Altman Z','Accruals','Screening Signal','Value Score','Quality Score','Risk Score','Overall Score','Custom Score','Custom Rank','Rank']
cols=[c for c in cols if c in scope.columns]
table=scope[cols].sort_values('Custom Score',ascending=False,na_position='last').copy()
table['fiscal_end']=table['fiscal_end'].dt.strftime('%Y-%m-%d')
st.dataframe(table,use_container_width=True,hide_index=True,height=510)
st.download_button('Download filtered table (CSV)',table.to_csv(index=False).encode('utf-8'),'industrial_equity_screen.csv','text/csv')

st.divider()
st.header('Financial chart playground')
st.caption('Charts use the selected sectors, score filters and BUY/SELL category. The full company table above stays visible.')
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

def metric_summary(row):
    fields = [('STA', 'STA'), ('SNOA', 'SNOA'), ('B/M', 'B/M'),
              ('GPA', 'GPA'), ('Altman Z', 'Altman Z')]
    return ', '.join(f'{label}: {row[key]:.4f}' if pd.notna(row[key])
                     else f'{label}: unavailable' for label, key in fields)

if signal == 'SELL Candidate':
    st.error(
        f"**SELL candidate.** This stock has a combined financial concern score of "
        f"{company['_SellStrength']:.1f}/100, meeting the selected SELL threshold of {sell_pct}. "
        "The screen evaluates accrual intensity, profitability, financial resilience, and valuation. "
        f"Company values: {metric_summary(company)}. "
        "These factors suggest further investigation of earnings quality and potential downside, "
        "not proof that the stock is overvalued."
    )
elif signal == 'BUY Candidate':
    st.success(
        f"**BUY candidate.** This stock has a combined financial opportunity score of "
        f"{company['_BuyStrength']:.1f}/100, meeting the selected BUY threshold of {buy_pct}. "
        "The screen rewards valuation and profitability, with additional support from "
        "earnings quality and financial resilience. "
        f"Company values: {metric_summary(company)}. "
        "These factors support a potential investment thesis, not a guarantee of future returns."
    )
else:
    sell_text = (f"{company['_SellStrength']:.1f}" if pd.notna(company['_SellStrength']) else 'unavailable')
    buy_text = (f"{company['_BuyStrength']:.1f}" if pd.notna(company['_BuyStrength']) else 'unavailable')
    st.info(
        f"**Neutral.** BUY screen: {buy_text}/100 (threshold {buy_pct}); "
        f"SELL screen: {sell_text}/100 (threshold {sell_pct}). "
        "The stock does not uniquely satisfy either complete screening rule, or some financial "
        f"data is unavailable. Company values: {metric_summary(company)}."
    )

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
st.header('Methodology')
st.markdown(f'''**How BUY and SELL candidates are selected**

The app compares manufacturing companies using five financial ratios: STA, SNOA, B/M, GPA, and Altman Z. Using several measures gives a broader picture than relying on just one ratio.

**SELL candidates** are screened mainly for high accruals, which can raise questions about earnings quality. Lower profitability, a lower Altman Z value, and weaker B/M also affect the result. A company qualifies when its SELL score reaches **{sell_pct}/100** and at least one accrual measure falls in the highest 40% for concern.

**BUY candidates** are screened mainly for stronger book to market values and gross profitability. Accruals and Altman Z also contribute. A company qualifies when its BUY score reaches **{buy_pct}/100** and both B/M and GPA rank above the 55th percentile. The BUY and SELL screens use different rules, rather than being exact opposites. Companies meeting both sets of rules, or neither, are marked **Neutral**. Missing financial data may also prevent a company from qualifying.

**Financial ratios used**

**STA:** Net income minus operating cash flow, divided by total assets. This is an overall accrual measure used as a proxy for short term accruals.

**SNOA:** A simplified measure of net operating assets relative to total assets.

**B/M:** Book equity divided by market capitalization. It compares a company's accounting value with its market value.

**GPA:** Gross profit divided by total assets. It measures gross profit relative to the company's asset base.

**Altman Z:** A measure used to assess financial distress risk.

**Accruals:** The average of STA and SNOA in this app. This is a custom measure, not a standard accounting ratio.

**How company rankings are calculated**

The ratios are converted into percentile scores so that companies can be compared on a common scale. Higher B/M, GPA, and Altman Z values are preferred, while lower STA and SNOA values are preferred. The **Custom Score** combines value, profitability, accrual quality, and financial resilience using the weights selected in the sidebar. Changing the weights updates the **Custom Rank**, while the original **Overall Score** and **Rank** saved in the CSV stay unchanged. BUY and SELL classifications are calculated separately from the Custom Score.

**Limitations**

The app uses annual financial data saved in `universe.csv`, not live prices or statements. Some companies have missing data. SNOA is simplified, and the Altman Z formula uses course coefficients, so standard bankruptcy cutoffs may not apply. The results are screening signals for further research, not guaranteed investment outcomes.''')
