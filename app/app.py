import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import silhouette_score

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🛍️ E-Commerce Customer Segmentation & RFM Dashboard")
st.markdown("This dashboard showcases customer behavior patterns, RFM metrics, and K-Means clustering segments.")

# 2. Load Data (Caching it so it loads fast)
@st.cache_data
def load_data():
    # Replace this with your final processed dataframe or CSV output from your notebook
    df=pd.read_csv(r'E:\dsProject\customerSegmentation\data\clean_Customer_RetailNw4.csv')
    # NOTE: Ideally, you load a pre-computed RFM dataframe with cluster labels assigned!
    return df
# data loading
df=load_data()

# 3. SIDEBAR FILTERS
st.sidebar.header('Filter Options')
# country_filter=st.sidebar.selectbox('Select Country',options=['All']+list(df['Country'].unique())) #country  NEED TO BE ADDED MORE THAN 1 COUNTRY
# COUNTRY fiLTER
country_filters=st.sidebar.multiselect(
    'Select Countries',
    options=list(df['Country'].unique()),
)
if country_filters:
    filtered_df=df[df['Country'].isin(country_filters)]
else:
    filtered_df=df.copy()

# /need to date range filter
# month_order = [
#     "January", "February", "March", "April", "May", "June",
#     "July", "August", "September", "October", "November", "December"
# ]# for sorting Month order
month_order=[
    'December(2009)',
    'January(2010)',
    'February(2010)',
    'March(2010)',
    'April(2010)',
    'May(2010)',
    'June(2010)',
    'July(2010)',
    'August(2010)',
    'September(2010)',
    'October(2010)',
    'November(2010)',
    'December(2010)'
]
available_months=[m for m in month_order if m in filtered_df['month'].unique() ] #for sorted Month name FIlter
months__filter=st.sidebar.multiselect(
    'Select Months',
    options=available_months
)
if months__filter:
    filtered_df=filtered_df[filtered_df['month'].isin(months__filter)]

# customer_segment=st.sidebar.select
segment_filter=st.sidebar.multiselect(
    'Select Customer Segments',
    options=list(filtered_df['ClusterLabel'].unique())
)
if segment_filter:
    filtered_df=filtered_df[filtered_df['ClusterLabel'].isin(segment_filter)]
# Note: On the DashBoard make sure maintain order while selecting filters,'cause it will present data on that manner.


st.html(
    """
    <style>
        /* Target the flex container holding the tab buttons */
        [data-testid="stTabs"] [role="tablist"] {
            justify-content: center;
            gap: 70px; 
        }
    </style>
    """
)
# tABS sETups
tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['ℹ️ About','📊 Analytics','🤖 RFM Clustering','Customer Look UP','ML CLustering','Building Process'])


with tab1:
    st.header("ℹ️ About This Dashboard")
    st.markdown(
       """
    Welcome to the **E-Commerce Customer Segmentation & RFM Dashboard**.
    
    This project is an **intelligent customer segmentation engine** that transforms raw, real-world retail transaction data into a targeted marketing playbook. The application is designed to showcase customer behavior patterns and RFM (Recency, Frequency, Monetary) metrics through actionable machine learning segments.
    
    By combining machine learning algorithms with targeted business logic, it analyzes customers' buying habits—specifically how recently they bought, how often they purchase, and how much they spend—and assigns them to distinct personas like 'Enterprise Whales' or 'At-Risk Customers.' 
    
    For a business, this tool solves the expensive problem of generic, "spray and pray" marketing. It takes the guesswork out of customer relationship management and provides a clear, data-driven roadmap to maximize marketing ROI.
    """
    )
    st.divider()

    st.subheader("💡 Dedicated Business Problems Solved")
    st.markdown("This engine directly tackles four of the most expensive challenges in retail and e-commerce:")
    with st.expander('🔍View the Core Business Problems',expanded=False):
        # Just two vertical tracks—cards will tightly stack with no vertical gaps
        prob1, prob2 = st.columns(2)

        with prob1:
            st.error(
                "**💸 Wasted Marketing Spend ('Spray and Pray')**\n\n"
                "**The Problem:** Sending blanket 20% discounts destroys profit margins because businesses end up discounting VIPs who were going to buy at full price anyway.\n\n"
                "**The Solution:** This engine categorizes users so marketing can target promotions *only* to those who actually need a financial incentive to convert."


            )
            st.warning(
                "**📉 Silent Customer Churn**\n\n"
                "**The Problem:** Companies rarely realize a customer has left until months after their last purchase, when it is too late to win them back.\n\n"
                "**The Solution:** By tracking Recency metrics, this project actively flags 'Hibernating' and 'At-Risk' users, enabling the deployment of automated retention emails before the customer permanently churns."
            )
            
        with prob2:
            st.success(
                "**🐋 Mismanaged VIP Relationships**\n\n"
                "**The Problem:** The top 1% of a retail customer base often drives up to 40% of the revenue. Treating these accounts like everyday shoppers risks losing massive accounts.\n\n"
                "**The Solution:** The algorithm mathematically isolates extreme monetary outliers so the business can assign dedicated account managers and provide white-glove, exclusive service."
            )
            st.info(
                "**🛒 Stagnant Average Order Value (AOV)**\n\n"
                "**The Problem:** Many customers buy frequently but only purchase cheap, single items, leaving money on the table.\n\n"
                "**The Solution:** By identifying 'Loyal Everyday Shoppers', the business knows exactly who to target with product bundles, cross-sells, or free-shipping thresholds to increase overall basket size."
            )

    # st.divider()

    st.subheader("💼 The Business Context")
    st.markdown("Understanding the *why* behind the data.")
    with st.expander("🔍 View The Core Business Insights (The 7 W's)", expanded=False):
            colA, colB = st.columns(2)
            
            with colA:
                st.write(
                    "**1. What business problem does it solve?**\n\n"
                    "Retailers bleed money by treating all customers equally. Sending a discount to a VIP wastes margin, while ignoring a fading loyalist guarantees churn."
                )
                st.write(
                    "**2. Why was this built?**\n\n"
                    "To replace static, intuition-based marketing with dynamic, data-driven customer targeting."
                )
                st.write(
                    "**3. Who is this for?**\n\n"
                    "Marketing Executives, CRM Managers, and E-commerce Strategists looking to optimize their campaign budgets."
                )
                st.write(
                    "**4. What data did you analyze?**\n\n"
                    "Historical retail transactional data encompassing hundreds of thousands of individual purchases, compressed into Recency, Frequency, and Monetary (RFM) behavioral profiles."
                )
    
            with colB:
                st.write(
                    "**5. What insights did you find?**\n\n"
                    "The customer base is not a monolith. The data organically revealed 6 distinct behavioral personas. Most notably, it isolated extreme 'VIP Whales' who require entirely different engagement tactics than 'Everyday Shoppers'."
                )
                st.write(
                    "**6. What business recommendations would you give?**\n\n"
                    "Deploy high-touch personal account managers for Whales, trigger automated cross-sell emails for Everyday Shoppers, and launch aggressive win-back discounts *only* for the At-Risk segment."
                )
                st.write(
                    "**7. What decision does it help someone make?**\n\n"
                    "It dictates exactly where to allocate the next dollar of marketing spend to achieve the absolute highest Return on Investment (ROI) and prevent revenue loss."
                )
    
    st.divider()
    
    st.subheader("🎯 Customer Segments")
    with st.expander(' 🔍View the Customer Segments from this business data',expanded=False):
        st.markdown(
            """
            The machine learning engine categorizes the customer base into six distinct operational groups:
            * 🐋 **Enterprise / VIP Whales:** High-monetary outliers requiring dedicated account management.
            * 🏆 **Champions/VIPs:** Top-tier reliable buyers; protect revenue margin and provide loyalty access.
            * 🛒 **Loyal Everyday Shoppers:** High-frequency buyers; maximize order size through targeted cross-selling.
            * 🆕 **New / Promising Customers:** Recent buyers; build recurring habits with onboarding flows.
            * 🛑 **Hibernating / Risk of Churn:** Fading buyers; re-engage via personalized high-value offers.
            * ⚠️ **At-Risk / Lost Customers:** Churned profiles; deploy dynamic win-back marketing before final churn.
            """
        )
    
    st.subheader("🛠️ Core Features")
    with st.expander('🔍 View the features from this application',expanded=False):
        st.markdown(
            """
            * **Business Analytics:** Track overarching KPIs like Total Revenue, Unique Customers, and Average Basket Value, alongside sales seasonality and geographic performance.
            * **RFM Clustering Details:** Analyze cluster centroids and view 2D/3D behavioral distribution maps.
            * **Operational Customer Lookup:** Search for individual profiles to extract raw transactional ledgers, anlyze individual customer pattern against training Data by using 3D behavioral distribution  and view micro-patterns (monthly, weekly, and hourly engagement).
            * **Machine Learning Simulator:** Input raw RFM values to instantly predict a customer's persona using the pre-trained `rfc.joblib` classification engine.
            * **Export Dataset:** This project helps you to export all Customer Segments of data / selective customer segments & even Individual Customer Segement Data for reachong out specific users.
            """
        )

    st.subheader("🔍 Strategic Insights & Recommendations")
    with st.expander('Provide Strategic Insights & Recommendations',expanded=False):
        st.markdown(
            """
            By mapping the customer base into a multi-dimensional behavioral space, several critical insights emerged:
            * **The Outlier Wealth:** A fraction of the user base ("Enterprise Whales") drives a massive percentage of overall revenue. **Recommendation:** Never blanket-discount this group. Assign them dedicated account managers and offer exclusive early-access perks.
            * **The Silent Churn:** Many formerly loyal shoppers fade away unnoticed. **Recommendation:** Deploy automated, high-incentive win-back campaigns specifically triggered when a user enters the "Hibernating" segment.
            * **The Mid-Tier Potential:** "Everyday Shoppers" visit frequently but spend little. **Recommendation:** Implement dynamic cross-selling and free-shipping thresholds to increase their Average Basket Value.
            """
        )

        st.success(
            """
            🎯 **The Ultimate Decision Engine:** 
            This dashboard helps business leaders make one crucial daily decision: **Who do we contact today, what do we say to them, and how much do we spend to retain them?**
            """
        )
    st.divider()
    st.subheader("🧠 Methodology: The 'Cluster-Then-Predict' Pipeline")
    st.markdown(
        """
        To ensure this system is both analytically rigorous and highly scalable for a production environment, it utilizes a two-stage hybrid machine learning architecture:
        * **Phase 1 - Discovery (K-Means & Heuristics):** Historical data was grouped using RFM (Recency, Frequency, Monetary) metrics. K-Means clustering, combined with business-logic outlier isolation, was used to discover mathematical boundaries and establish baseline ground-truth personas.
        * **Phase 2 - Deployment (Random Forest):** A Random Forest Classifier (RFC) was trained on the discovered segments. Because tree-based models natively handle extreme outliers and non-linear boundaries, this pipeline bypasses complex feature scaling. It processes raw RFM inputs to deliver lightning-fast, highly interpretable predictions.
        """
    )
    st.subheader("💻 Tech Stack")
    st.markdown(
        """
        * **Frontend:** Streamlit
        * **Data Processing:** Pandas, NumPy
        * **Machine Learning:** Scikit-Learn (K-Means, Silhouette Score,Random Forest Classifier )
        * **Visualizations:** Plotly Express
        """
    )
    st.divider()

    st.subheader("📌 Project Repo & Contact")

    # Creating four clean, equal columns for a modern footer layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**📂 Dataset Source**\n\n[UCI Machine Learning Repo](https://archive.ics.uci.edu/dataset/502/online+retail+ii)")

    with col2:
        st.markdown("**💻 Source Code**\n\n[GitHub Repository](https://github.com)")

    with col3:
        st.markdown("**📬 Email Inquiries**\n\n mannk7062@gmail.com")

    with col4:
        st.markdown("**🤝 Professional Network**\n\n[LinkedIn Profile](http://www.linkedin.com/in/mann-32718a1b9)")


with tab2:
    st.header('📈 Business Analytics')
    st.subheader('Key Performance Indicators')

    # KPI Ribbon
    kpi1,kpi2,kpi3,kpi4=st.columns(4)

    # # Caluclating Delta : value_count of every month->% every month store in a storted manner in a list ->Compare last 2
    # total_Earning=filtered_df['TotalSum'].sum()
    # Del_monthly_sales=(filtered_df.groupby('month').agg(Monetary=('TotalSum','sum')).assign(RevPercent=lambda x:(x['Monetary']/total_Earning * 100).round(2)).reindex(month_order)).reset_index()
    total_Earning=filtered_df['TotalSum'].sum()
    total_frequency=filtered_df['Frequency'].sum()
    total_users=filtered_df['Customer ID'].nunique()

    Del_monthly_sales=(filtered_df.groupby('month').agg(Monetary=('TotalSum','sum'),Frequency=('Frequency','sum'),Users=('Customer ID','nunique'))
                .assign(FreqPercent=lambda x:(x['Frequency']/total_frequency * 100).round(2),RevPercent=lambda x:(x['Monetary']/total_Earning * 100).round(2)
                        ,UsrPercent=lambda x:(x['Users']/total_users * 100).round(2))).reindex(month_order).dropna().reset_index()
    def deltaGen_Rev(data=Del_monthly_sales):
        valid_data=Del_monthly_sales.dropna()

        if len(valid_data)<2:
            return
        lst=list(valid_data['RevPercent'].iloc[-2:])
        diff=round(lst[1]-lst[0],2)
        if diff==0:
            return f'{diff}% = Last Mo'
        elif diff>0:
            return f'{diff}% vs Last Mo'
        else:
            return f'{diff}% vs Last Mo'
        
    def deltaGen_freq(data=Del_monthly_sales):
            valid_data=Del_monthly_sales.dropna()
    
            if len(valid_data)<2:
                return
            lst=list(valid_data['FreqPercent'].iloc[-2:])
            diff=round(lst[1]-lst[0],2)
            if diff==0:
                return f'{diff}% = Last Mo'
            elif diff>0:
                return f'{diff}% vs Last Mo'
            else:
                return f'{diff}% vs Last Mo'

    def deltaGen_usr(data=Del_monthly_sales):
                valid_data=Del_monthly_sales.dropna()
        
                if len(valid_data)<2:
                    return
                lst=list(valid_data['UsrPercent'].iloc[-2:])
                diff=round(lst[1]-lst[0],2)
                if diff==0:
                    return f'{diff}% = Last Mo'
                elif diff>0:
                    return f'{diff}% vs Last Mo'
                else:
                    return f'{diff}% vs Last Mo'
    


    kpi1.metric('💰 Total Revenue',f'${filtered_df['TotalSum'].sum():,.2f}',delta=f'{deltaGen_Rev()}' if deltaGen_Rev() else None)
    kpi2.metric('📦 Total Frequency',f'Σ {filtered_df['Invoice'].nunique():,}',delta=f'{deltaGen_freq()}' if deltaGen_freq() else None)
    kpi3.metric('👥 Unique customers',f'Σ {filtered_df['Customer ID'].nunique():,}',delta=f'{deltaGen_usr()}'if deltaGen_usr() else None)
    kpi4.metric('🛒 Avg Basket Value',f'${filtered_df['TotalSum'].mean():,.2f}')

    st.divider()
#   Sales Seasonality trend
    # Need to add seletive filter 'country'  above this chart
    st.subheader('📈 Sales Seasonality & Trends')
    col_time1,col_time2=st.columns(2)

    with col_time1: # need to replace bar plot to line plot
        st.markdown("<p style='text-align: center;'><b>📅 Monthly Revenue Trend (1Dec2009 to 9Dec2010)</b></p>", unsafe_allow_html=True)

        fig_monthly=px.line(Del_monthly_sales,x='month',y='Monetary',markers=True,line_shape='linear',color_discrete_sequence=['#457b9d'],hover_data={'RevPercent':True,'FreqPercent':True,'UsrPercent':True})
        fig_monthly.update_layout(xaxis_title='Month',yaxis_title='Revenue($)')
        fig_monthly.update_xaxes(categoryorder='array',categoryarray=month_order)
        st.plotly_chart(fig_monthly,use_container_width=True)

    with col_time2: #need to replace bar plot to line plot
        st.markdown("<p style='text-align: center;'><b>🛍️ Busiest Shopping Days (1Dec2009 to 9Dec2010)</b></p>", unsafe_allow_html=True)
        weekDay_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        daily_sales=filtered_df.groupby('weeks')['Invoice'].nunique().reindex(weekDay_order).dropna().reset_index()
        fig_day=px.line(daily_sales,x='weeks',y='Invoice',markers=True,line_shape='linear',color_discrete_sequence=["#459d51"])
        fig_day.update_layout(xaxis_title='Week of Year/Day Profile',yaxis_title='Order Volume')
        st.plotly_chart(fig_day,use_container_width=True)

        # think about addding hourly analytics graph

    st.divider()
# Geographic Performance
    # Need to add seletive filter 'country'  above this chart
    st.subheader('🌍 Top 10 Performing Region')
    country_sales=filtered_df.groupby('Country')['TotalSum'].sum().reset_index().sort_values(by='TotalSum',ascending=False).head(10)
    fig_country=px.bar(country_sales,x='TotalSum',y='Country',orientation='h',color='TotalSum',color_continuous_scale='Blues')
    fig_country.update_layout(xaxis_title='Total Revenue($)',yaxis_title='Countries',yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_country,use_container_width=True)

    st.divider()

    # Present Customer Segment from data
    st.subheader("🚀 Customer Value Segment Breakdown")

    st.markdown('Segment Financial Snapshot')
    st.caption("Juxtaposition of specific user base volume against corporate profit contribution.")
    segment_summary=filtered_df.groupby('ClusterLabel').agg(
    Customer_count=('Customer ID','nunique'),
    Revenue=('TotalSum','sum')).reset_index()

    total_cust=segment_summary['Customer_count'].sum()
    total_rev=segment_summary['Revenue'].sum()

    segment_summary['Customer_share(%)']=(segment_summary['Customer_count']/total_cust)*100
    segment_summary['Revenue_share(%)']=(segment_summary['Revenue']/total_rev)*100
    # melting data for visualization
    df_melt=segment_summary.melt(id_vars='ClusterLabel',value_vars=['Customer_share(%)','Revenue_share(%)'],var_name='Corporate Metric',value_name='Percentage Value')
    # Bar Chart
    fig=px.bar(
        df_melt,
        x="ClusterLabel",
        y="Percentage Value",
        color="Corporate Metric",
        barmode="group",
        color_discrete_sequence=["#a8dadc", "#457b9d"], # Executive slate colour palette
        text=df_melt["Percentage Value"].map("{:.1f}%".format)
    )
    fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Percent Impact (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=380
        )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


    # Present Customer Segment  Value Counts
    st.markdown('**Prespective ROI Playbook**')
    st.caption("Strategic operational initiatives immediately assigned to distinct target groups.")

    actions = {
    "New / Promising Customers": "Build recurring habits. Trigger targeted welcome and onboarding email flows.",
    "Champions/VIPs": "Protect revenue margin. Provide early loyalty access and premium perks.",
    "At-Risk / Lost Customers": "Deploy dynamic automated win-back marketing emails before final churn.",
    "Hibernating / Risk of Churn": "Re-engage via high-incentive multichannel SMS and personalized offers.",
    "Loyal Everyday Shoppers": "Maximize order size. Cross-sell related product categories and bundles.",
    "Enterprise / VIP Whales": "Crucial accounts. Assign personal account managers. Do not discount."
    }
    segment_summary['Strategic_action']=segment_summary['ClusterLabel'].map(actions)

    df_report=segment_summary.copy()
    df_report['Segment size']=df_report['Customer_count'].map("{:,} users".format)
    df_report['Financial yield']=df_report['Revenue'].map("{:,.2f}".format)
    # clean structure DF
    df_report=df_report[['ClusterLabel','Segment size','Financial yield','Strategic_action']]

    st.dataframe(
        df_report,
        column_config={
            "ClusterLabel": st.column_config.TextColumn("Customer Segment Profile",width='medium'),
            "Segment size": st.column_config.TextColumn('Total Volume',width='small'),
            "Financial yield": st.column_config.TextColumn('Revenue Driven',width='small'),
            "Strategic_action": st.column_config.TextColumn('Executive Strategic Action Path',width='large')
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('---')

# with tab3:
with tab3:
    st.header("🤖 RFM Clustering Distribution Details")
    st.markdown("Use this tab to analyze how Recency, Frequency, and Monetary numerical scales interact across clusters.")

    st.subheader("📊 Segmentation Model Validation")
    # Creating a ribbon
    val1,val2,val3=st.columns(3)

    with val1:
        st.metric(label='🧩 Optimal Cluster(K)',value=f'{filtered_df['ClusterLabel'].nunique()}')

    with val2:
        st.metric(label='📉 Model Silhouette Score',value='0.40',help='Scores closer to 1 mean cluster are well seperated.')
    
    with val3:
        st.metric(label='✅ Business-Adjusted Accuracy',value='98%',help='All active database records successfully assigned to a cluster matrix')

    st.divider()
    st.subheader("📐 Cluster Centroids (Behavioral Averages)")
    st.caption("This table reveals the exact mathematical averages that define each customer profile group.")

    # Calculate the mean for each cluster dynamically based on filters
    cluster_centroids = filtered_df.groupby('ClusterLabel').agg(
                Avg_Recency=('Recency', 'mean'),
                Avg_Frequency=('Frequency', 'mean'),
                Avg_Monetary=('TotalSum', 'mean'),
                Total_Customers=('Customer ID', 'nunique')
            ).reset_index()
        
    st.dataframe(
        cluster_centroids,
        column_config={
            "ClusterLabel": st.column_config.TextColumn("Customer Segment Profile"),
            "Avg_Recency": st.column_config.NumberColumn("Avg Recency (Days)", format="%.1f days"),
            "Avg_Frequency": st.column_config.NumberColumn("Avg Frequency (Orders)", format="%.1f orders"),
            "Avg_Monetary": st.column_config.NumberColumn("Avg Monetary (Spend)", format="$%,.2f"),
            "Total_Customers": st.column_config.NumberColumn("Total Users", format="%,d")
        },
        hide_index=True,
        use_container_width=True
    )


    st.divider()

    # Bubble Grid Chart
    # Fetching only RFM DATA
    rfm_df = filtered_df.drop_duplicates(subset=['Customer ID'])[['Customer ID', 'Recency', 'Frequency', 'MonetaryValue','ClusterLabel']].reset_index(drop=True)

    fig_bubble=px.scatter(
        rfm_df,
        x='Recency',
        y='Frequency',
        size='MonetaryValue',
        color='ClusterLabel',
        log_y=True,
        title='2D RFM Cluster Distribution Map(Size=Monetary Value)',
        size_max=35,# Prevents massive whale bubbles from cluttering screen
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_bubble.update_layout(
        height=650,
        margin=dict(l=60,r=40,b=100,t=60),
        xaxis=dict(
            title_text='Recency (Days SInce last order)',
            title_font=dict(size=14,color='black'),
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title_text='Frequency (Total Tansactions - Log Scale)',
            title_font=dict(size=14,color='black'),
            tickfont=dict(size=11)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5
        )
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.divider()

    st.subheader("📥 Exporting Customer Segment Data")
    st.caption("You can export Customer Segments Data or Individual Customer Segment in a .CSV Format.")

    all_segments=list(filtered_df['ClusterLabel'].unique())
    tar_sgmnt=st.multiselect('Choose Segment to Export',key='Choose one or multiple Segents',options=all_segments)

    if len(tar_sgmnt)>0:

        exprt_df=filtered_df[filtered_df['ClusterLabel'].isin(tar_sgmnt)]

        st.success(f'Selected **{len(tar_sgmnt)}** segments matching a total of **{len(exprt_df):,}** raw data records.')

        csv_bytes=exprt_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label=f'📥 Download Export file {len(exprt_df):,} rows',
            data=csv_bytes,
            file_name=f'CustomerSegmentDF.csv',
            mime='text/csv',
            use_container_width=True
        )

# Individuatl Customer Lookup
with tab4:
    st.header('🔍 Operational Customer Lookup Hub')
    st.markdown("Extract targeting files or research individual profiles for marketing outreach campaigns.")
    
    look1, look2 = st.columns([1,2])
    
    with look1:
        st.subheader("Profile Target Investigation")
        available_cust = filtered_df['Customer ID'].dropna().unique()
        if len(available_cust) > 0:
            selected_cust = st.selectbox("Search/Select Customer ID", options=available_cust)
            # filtering customer ID
            cust_data = filtered_df[filtered_df['Customer ID'] == selected_cust]
            
            st.markdown(f"**Assigned Segment Profile:** `{cust_data['ClusterLabel'].iloc[0] if not cust_data.empty else 'N/A'}`")

            kpi1,kpi2=st.columns(2)
            with kpi1:
                st.metric("Total Revenue Contribution", f"${cust_data['TotalSum'].sum():,.2f}")
                st.metric('Days Since Last Order',f'{cust_data['Recency'].min()} Days')

            with kpi2:
                st.metric("Total Order Volume", f"{cust_data['Invoice'].nunique():,}")
                # percetile standing calculation
                cust_total=filtered_df.groupby('Customer ID').agg(Monetary=('MonetaryValue','max')).reset_index()
                cust_total['Percentile']=cust_total['Monetary'].rank(pct=True)*100
                target_pct=cust_total[cust_total['Customer ID']==selected_cust]['Percentile'].iloc[0]
                st.metric('Percentile Standing',f'{target_pct:.1f}th')

    with look2:
        # last top5 purchased items showing
        st.markdown('**Recent Items Purchased:**',text_alignment='center')
        top_items=cust_data[['Description','Quantity','InvoiceDate']].tail(8)
        st.dataframe(
                    top_items,
                    column_config={
                        'Description':st.column_config.TextColumn('Items Name',width='medium',alignment='center'),
                        'Quantity':st.column_config.NumberColumn('Quantity',width=5,alignment='center'),
                        'InvoiceDate':st.column_config.NumberColumn('Invoice Date',width='small',alignment='center')
                    },
                    use_container_width=True,
                    hide_index=True
                )

    st.divider() # Adds a clean visual line

    st.subheader('Visualization pattern',text_alignment='center')
    # st.caption('Visualization : Locted on 3D plot, Months ,week days, Hour ')

    # ==========================================
    # 🎨 PROFESSIONAL LAYOUT & THEME CONFIGURATION
    # ==========================================
    THEME_ACCENT = "#2A9D8F"       # Deep Emerald Teal for standard trends
    THEME_TARGET = "#E63946"       # Vivid Coral-Red for targeted profile highlighting
    THEME_BASE = "#555656"         # Muted, clean light slate gray for base populations
    PLOT_TEMPLATE = "plotly_white"  # Premium minimal grid background

    # ==========================================
    # 🏢 ROW 1: MACRO VIEW (3D COHORT MAPPING)
    # ==========================================
    st.markdown(f"##### 📌 Macro Cohort Placement")
    
    # Inject classification labels and trace sorting priorities
    rfm_df['VisualizeHighlight'] = rfm_df['Customer ID'].apply(
        lambda x: '🎯 Selected Profile' if x == selected_cust else '👥 Base Population'
    )
    rfm_df['MarkerSize'] = rfm_df['Customer ID'].apply(
        lambda x: 22 if x == selected_cust else 4
    )
    
    # Sort data frame so the highlighted user draws last (forces it on top of the visual stack)
    rfm_sorted = rfm_df.sort_values(by='VisualizeHighlight', ascending=False)

    fig_3d = px.scatter_3d(
        rfm_sorted, x='Recency', y='Frequency', z='MonetaryValue',
        size='MarkerSize',
        color='VisualizeHighlight',
        color_discrete_map={'🎯 Selected Profile': THEME_TARGET, "👥 Base Population": THEME_BASE},
        hover_data={
            'Customer ID': True, 'Recency': True, 'Frequency': True, 
            'MonetaryValue': ':$,.2f', 'ClusterLabel': True, 
            'VisualizeHighlight': False, 'MarkerSize': False
        },
        opacity=0.75,
        size_max=22, 
        template=PLOT_TEMPLATE
    )
    
    fig_3d.update_traces(marker=dict(line=dict(width=1.5, color='white'), symbol='circle'))
    
    fig_3d.update_layout(
        title=dict(text=f"<b>3D Space Localization: Customer {selected_cust} vs Market Base</b>", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=0.92, xanchor="right", x=1, title_text=""),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(title='Recency (Days)', gridcolor='#F3F4F6', backgroundcolor='#FAFAFA'),
            yaxis=dict(title='Frequency (Orders)', gridcolor='#F3F4F6', backgroundcolor='#FAFAFA'),
            zaxis=dict(title='Monetary Value ($)', gridcolor='#F3F4F6', backgroundcolor='#FAFAFA'),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # ==========================================
    # 🕒 ROW 2: MICRO VIEW (TEMPORAL DRILLDOWN TRENDS)
    # ==========================================
    st.markdown("##### 🕒 Granular Purchase Micro-Patterns")
    vis_month, vis_week, vis_hour = st.columns(3)

    # --- COLUMN 1: MONTHLY LIFE CYCLE ---
    with vis_month:
        st.markdown(f"📊 **Monthly Revenue Trajectory**")

        spending_mnth = cust_data.groupby('month').agg(
            Spending=('TotalSum', 'sum'),
            Frequency=('month', 'count'),
            Visiting=('InvoiceDate', 'nunique')
        ).reindex(month_order).dropna().reset_index()

        spend_figMnth = px.area(
            spending_mnth, x='month', y='Spending', markers=True,
            color_discrete_sequence=[THEME_ACCENT],
            hover_data={'Frequency': True, 'Visiting': True},
            template=PLOT_TEMPLATE
        )
        spend_figMnth.update_layout(margin=dict(l=10, r=10, b=10, t=10), xaxis_title="", yaxis_title="Spending ($)", hovermode="x unified")
        spend_figMnth.update_xaxes(categoryorder='array', categoryarray=month_order)
        st.plotly_chart(spend_figMnth, use_container_width=True)

    # --- COLUMN 2: WEEKLY VELOCITY ---
    with vis_week:
        st.markdown(f"🗓️ **Weekly Spending Velocity**")
        
        spending_wks = cust_data.groupby('weeks').agg(
            Spending=('TotalSum', 'sum'),
            Frequency=('weeks', 'count'),
            Visiting=('InvoiceDate', 'nunique'),
            Month=('month','unique')
        ).reindex(weekDay_order).dropna().reset_index()
        
        spending_wks_fig = px.area(
            spending_wks, x='weeks', y='Spending', markers=True,
            color_discrete_sequence=[THEME_ACCENT],
            hover_data={'Frequency': True, 'Visiting': True,'Month':True},
            template=PLOT_TEMPLATE
        )
        spending_wks_fig.update_layout(margin=dict(l=10, r=10, b=10, t=10), xaxis_title="", yaxis_title="Spending ($)", hovermode="x unified")
        spending_wks_fig.update_xaxes(categoryorder='array', categoryarray=weekDay_order)
        st.plotly_chart(spending_wks_fig, use_container_width=True)

    # --- COLUMN 3: HOURLY ENGAGEMENT ---
    with vis_hour:
        st.markdown(f"⏰ **Hourly Engagement Grid**")
        
        spending_hours = cust_data.groupby('hours').agg(
            Spending=('TotalSum', 'sum'),
            Frequency=('Invoice', 'count'), 
            Visiting=('InvoiceDate', 'nunique'),
            Month=('month','unique'),
            Week=('weeks','unique')
        ).dropna().reset_index()
        
        spending_hourFig = px.area(
            spending_hours, x='hours', y='Spending', markers=True,
            color_discrete_sequence=[THEME_ACCENT],
            hover_data={'Frequency': True, 'Visiting': True, 'Month':True ,'Week':True},
            template=PLOT_TEMPLATE
        )
        spending_hourFig.update_layout(margin=dict(l=10, r=10, b=10, t=10), xaxis_title="Hour of Day (24h)", yaxis_title="Spending ($)", hovermode="x unified")
        spending_hourFig.update_xaxes(autorange=True, dtick=4) # Standardized tick spacing for narrow columns
        st.plotly_chart(spending_hourFig, use_container_width=True)

        
                # assigning Marker Size on gra
    st.divider() # Adds a clean visual line


    st.subheader("🎯 Recommended Marketing Strategy")

    # Get the customer's segment name safely
    if not cust_data.empty:
        current_segment = str(cust_data['ClusterLabel'].iloc[0]).strip() 
        
        # Create a dictionary of strategy rules
        strategies = {
                        "New / Promising Customers": "🚀 **Strategy:** Welcome sequence. Send helpful guides and a short-term discount to trigger a second purchase quickly.",
                        "Champions/VIPs": "🏆 **Strategy:** Reward them. Offer early access to new products or a VIP loyalty program. Do not discount heavily; they already love you.",
                        "At-Risk / Lost Customers": "🚨 **Strategy:** Last-chance win back. Use aggressive discounts or automated emails. Clean them from lists if they remain inactive.",
                        "Hibernating / Risk of Churn": "💤 **Strategy:** Re-engage them. Send 'We Miss You' discounts and showcase your newest product arrivals.",
                        "Loyal Everyday Shoppers": "🛒 **Strategy:** Increase order size. Offer product bundles or a free-shipping threshold to get them to spend more per visit.",
                        "Enterprise / VIP Whales": "🐋 **Strategy:** High-touch care. Provide personal account outreach, premium perks, or custom bulk packages."
                    }

        
        # Get the strategy or show a default message if the name doesn't match perfectly
        action_plan = strategies.get(current_segment, "💡 **Strategy:** Analyze general behavior and offer standard email engagement newsletters.")
        
        # Display it in a nice colored informational box
        st.info(action_plan)
    
        st.divider()

        st.subheader("📥 Export Individual Profile Records")
        
        # 1. Cleanly extract individual string values to avoid bracket printing issues [12345]
        current_id = str(cust_data['Customer ID'].iloc[0]) if not cust_data.empty else "N/A"
        current_label = str(cust_data['ClusterLabel'].iloc[0]) if not cust_data.empty else "N/A"
        total_records = len(cust_data)

        st.markdown(f"Generate and extract a raw transactional archive for Customer ID: **{current_id}**.")

        # 2. Prepare the dataset for downloading
        csv_bytes = cust_data.to_csv(index=False).encode('utf-8')

        # 3. Create a clean dashboard container layout
        with st.container(border=True):
            col1, col2 = st.columns([3, 1], vertical_alignment="center") # Allocates more room for text
            
            with col1:
                st.markdown(
                    f"**System Status:** Ready to package **{total_records:,}** historical logs for "
                    f"Customer ID `{current_id}` (Segment: :blue[{current_label}]) :"
                )
                
            with col2:
                st.download_button(
                    label=f"Extract Ledger (CSV)",
                    data=csv_bytes,
                    file_name=f"customer_profile_{current_id}.csv",
                    mime='text/csv',
                    use_container_width=True, # Looks much cleaner inside a set column width
                    type="primary" # Gives it a professional accent color highlight
                )

# # Clustering Tab5
# Ensure this script has access to your artifacts directory
with tab5:
    st.header("🤖 Machine Learning Customer Clustering")
    st.markdown(
        """
        Predict customer personas instantly using an unsupervised **K-Means Clustering** engine. 
        The system dynamically screens for high-value outliers before passing behavioral metrics 
        through log transformation and standardisation scalers.
        """
    )
    st.markdown("---")

    # Layout: Align input fields and action button perfectly along the bottom axis
    col_r, col_f, col_m, col_btn = st.columns(4, vertical_alignment="bottom")

    with col_r:
        user_r = st.number_input(
            label="⏱️ Recency (Days since last purchase)",
            min_value=0,
            max_value=450,
            value=1,
            step=1,
            help="Number of days since the customer last placed an order."
        )

    with col_f:
        user_f = st.number_input(
            label="🔄 Frequency (Total lifetime visits)",
            min_value=0,
            max_value=365,
            value=1,
            step=1,
            help="Total number of discrete transactions or sessions recorded."
        )

    with col_m:
        user_m = st.number_input(
            label="💰 Monetary Value (Total Spend $)",
            min_value=0.00,
            max_value=300000.00,
            value=1.00,
            step=10.00,
            format="%.2f",
            help="Aggregate gross revenue generated by this customer."
        )

    with col_btn:
        predict_clicked = st.button(
            label="Run Behavior Analysis",
            type="primary",
            use_container_width=True
        )

    # Execution Flow on Trigger
    if predict_clicked:
        # CLUSTER_LABELS = {
        #                 'New / Promising Customers': "🆕 New / Recent Customers",
        #                 'Champions/VIPs': "🏆 VIPs & Champions",
        #                 'At-Risk / Lost Customers': "⚠️ Lost / Churned Customers",
        #                 'Hibernating / Risk of Churn': "🛑 Hibernating / Risk of Churn",
        #                 'Loyal Everyday Shoppers': "🛒 Loyal Everyday Shoppers",
        #                 'Enterprise / VIP Whales': "🐋 Enterprise / VIP Whales"
        #             }
        try:
            # 1. Load pipeline artifacts securely
            rfcSegment=joblib.load(r'artifacts\RFCsegmentaion.joblib')
            Customer_Segmentation=rfcSegment.predict([[user_r,user_f,user_m]])[0]
            st.markdown(f'{Customer_Segmentation}')

            # 2. Unified Persona Dictionary
            CLUSTER_LABELS = {
                'New / Promising Customers': "🆕 New / Recent Customers",
                'Champions/VIPs': "🏆 VIPs & Champions",
                'At-Risk / Lost Customers': "⚠️ Lost / Churned Customers",
                'Hibernating / Risk of Churn': "🛑 Hibernating / Risk of Churn",
                'Loyal Everyday Shoppers': "🛒 Loyal Everyday Shoppers",
                'Enterprise / VIP Whales': "🐋 Enterprise / VIP Whales"
            }

            st.markdown("### 📊 Segmentation Diagnostics")
            
            with st.container(border=True):
                # 3. Rule-Based Outlier Evaluation Phase

                if Customer_Segmentation=='Enterprise / VIP Whales':

                    st.success(f"**Segment Assigned:** {CLUSTER_LABELS[Customer_Segmentation]}")
                    st.markdown(
                            """
                            💡 **Strategic Playbook — Premium White-Glove Retention:**
                            * **Action:** Assign a dedicated VIP Account Management representative for direct communication.
                            * **Exclusivity:** Provide early access to new product lines and invitation-only enterprise events.
                            * **Service:** Upgrade account defaults to automated priority shipping and immediate support routing.
                            """
                        )
                    
                elif Customer_Segmentation=='Hibernating / Risk of Churn':

                    st.warning(f"**Segment Assigned:** {CLUSTER_LABELS[Customer_Segmentation]}")
                    st.markdown(
                            """
                            💡 **Strategic Playbook — High-Value Revenue Rescue:**
                            * **Action:** Deploy direct outreach from executive or customer success leadership to rebuild rapport.
                            * **Incentive:** Offer premium renewals, custom contract renegotiations, or deep tailored discounts.
                            * **Investigation:** Trigger an internal account health review to find and resolve hidden service friction points.
                            """
                        )
                    
                elif Customer_Segmentation=='Loyal Everyday Shoppers':
                    st.success(f"**Segment Assigned:** {CLUSTER_LABELS[Customer_Segmentation]}")
                    st.markdown(
                            """
                            💡 **Strategic Playbook — High-Frequency Monetisation:**
                            * **Action:** Launch cross-selling initiatives featuring higher-margin items based on past trends.
                            * **Upsell:** Introduce a loyalty subscription program to lock in long-term transactional value.
                            * **Engagement:** Use gamified milestone achievements or tiered badges to sustain high engagement levels.
                            """
                        )

                elif Customer_Segmentation=='Champions/VIPs':
                    st.success(f"**Segment Assigned:** {CLUSTER_LABELS[Customer_Segmentation]}")
                    st.markdown(
                        """
                        💡 **Strategic Playbook — Maximise Lifetime Value (LTV):**
                        * **Action:** Enroll them into an elite, tier-based VIP rewards program.
                        * **Upsell:** Pitch premium subscription tiers or high-margin product bundles.
                        * **Advocacy:** Leverage them for user-generated content, reviews, and referral milestones.
                        """
                        )
                                    
                elif Customer_Segmentation=='New / Promising Customers':
                    st.success(f"**Segment Assigned:** {CLUSTER_LABELS[Customer_Segmentation]}")
                    st.markdown(
                        """
                        💡 **Strategic Playbook — Nurture & Convert:**
                        * **Action:** Deploy automated 'Welcome Series' email drips with clear onboarding paths.
                        * **Incentive:** Offer a small discount or free shipping on their second purchase within 14 days.
                        * **Education:** Send content highlighting product utility, FAQs, and top-rated customer favorites.
                        """
                        )

                else:
                    st.error(f"**Segment Assigned:** {CLUSTER_LABELS[Customer_Segmentation]}")
                    st.markdown(
                        """
                        💡 **Strategic Playbook — Win-Back & Reactivation:**
                        * **Action:** Launch an aggressive 'We Miss You' discount campaign (e.g., 20% off site-wide).
                        * **Feedback:** Send a single-question survey to uncover why they stopped purchasing.
                        * **Cost Control:** Limit paid ad spend retargeting this group if they fail to reactivate within 30 days.
                        """
                    )
                        
        except FileNotFoundError as e:
            st.error(r"🚨 **System Error:** Model artifacts directory missing. Ensure standard `.joblib` files exist in the `.\artifacts\` directory.")
        except Exception as e:
            st.error(f"🚨 **Execution Error:** {str(e)}")
        st.divider()
        st.subheader('🌐 Live User Position vs. Trained Historical Customer Base')

        user_df=pd.DataFrame({'Recency': user_r,
                     'Frequency': user_f,
                     'MonetaryValue': user_m,
                     'ClusterLabel':f'Live User:{CLUSTER_LABELS[Customer_Segmentation]}',
                     'MarkerSize':[35]})
        # rfm_df
        # rfm_df = filtered_df.drop_duplicates(subset=['Customer ID'])[['Customer ID', 'Recency', 'Frequency', 'MonetaryValue','ClusterLabel']].reset_index(drop=True)
        live_rfmDF=rfm_df[['Recency', 'Frequency', 'MonetaryValue','ClusterLabel']] #SOURCE data (RFM)  extractING
        live_rfmDF['MarkerSize']=5
        # concat trained + Live user data
        liveRFMdF=pd.concat([live_rfmDF,user_df],ignore_index=True)

        color_map = {
                    "New/Recent Customers": "#1F77B4",         # Blue
                    "VIPs,Champions": "#2CA02C",               # Green
                    "Lost/Churned Customers": "#D62728",         # Red
                    "Hibernating / Risk of Churn": "#FF520E",    # Orange
                    "Loyal Everyday Shoppers ": "#9467BD",        # Purple
                    "Enterprise / VIP Whales": "#8C564B",         # Brown
                    user_df['ClusterLabel'].iloc[0]: "#00FFFF"      # Neon Cyan Highlighter for Live Point
                }

        # Visualiing LIVE USER DATA Against trined data
        figLiveUser=px.scatter_3d(liveRFMdF,x='Recency',y='Frequency',z='MonetaryValue',
                          size='MarkerSize',size_max=35,color='ClusterLabel',color_discrete_map=color_map,
                          opacity=0.7,labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Visits)', 'Monetary': 'Monetary ($)'},
                    title="Behavioral Space Mapping (3D Multi-Dimensional View)")
        figLiveUser.update_layout(
                    margin=dict(l=0, r=0, b=0, t=40),
                    legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
                    scene=dict(
                        xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", showbackground=True),
                        yaxis=dict(backgroundcolor="rgb(220, 220, 220)", gridcolor="white", showbackground=True),
                        zaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True),
                    ),
                    height=450
                )
        st.plotly_chart(figLiveUser,use_container_width=True)

with tab6:
    st.header("⚙️ System Architecture & Building Process")
    st.markdown(
        """
        This project utilizes an advanced **Cluster-Then-Predict** machine learning pipeline. 
        It bridges the gap between exploratory unsupervised learning and a low-latency production backend, ensuring scalable and robust customer classification.
        """
    )
    
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1️⃣ Discovery Phase")
        st.markdown(
            """
            * **Data Ingestion:** Processed raw 2009-2010 retail transaction logs to establish a clean dataset.
            * **RFM Aggregation:** Rolled up transaction-level data into customer-level Recency, Frequency, and Monetary features.
            * **Heuristic Pseudo-Labeling:** Applied K-Means clustering alongside visual business-logic outlier isolation to assign six distinct ground-truth labels.
            """
        )

    with col2:
        st.subheader("2️⃣ Supervised Training")
        st.markdown(
            """
            * **Raw Feature Utilization:** Bypassed scaling and transformations, leveraging the native outlier-immunity of tree-based algorithms.
            * **Model Optimization:** Trained a Random Forest Classifier (RFC) on the pseudo-labeled dataset to learn complex, non-linear segment boundaries.
            * **Artifact Serialization:** Exported the trained RFC and scaler into deployment-ready `.joblib` files.
            """
        )

    with col3:
        st.subheader("3️⃣ Dashboard Deployment")
        st.markdown(
            """
            * **Frontend Interface:** Built a responsive web application using Streamlit and Plotly for macro-level KPIs and 3D data visualization.
            * **Live Inference Engine:** Captures new customer RFM inputs via the UI, scales them, and passes them to the RFC artifact for real-time persona classification.
            """
        )

    st.divider()
    
    st.markdown("<p style='text-align: center;'><b>🔄 End-to-End System Flow</b></p>", unsafe_allow_html=True)
    st.code(
        '''
        [Raw Transactions] ➔ [RFM Feature Engineering] ➔ [K-Means + Visual Heuristics] ➔ [Labeled Dataset]
                                                                                                ↓
                             [Model Artifacts: rfc.joblib] ⟵ [Random Forest Classifier] ⟵ [Labeled Dataset]
                                          ↓
        [Streamlit UI] ➔ [Live RFM Input] ➔ [RFC Inference Engine] ➔ [Targeted Marketing Strategy]
        ''', 
        language='text'
    )