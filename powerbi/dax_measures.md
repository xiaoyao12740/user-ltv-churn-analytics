# DAX measures

```DAX
Total Users = DISTINCTCOUNT(powerbi_users[user_id])
Revenue = SUM(monthly_kpis[revenue])
MAU = SUM(monthly_kpis[mau])
ARPU = DIVIDE([Revenue], [MAU])
Paying Users = SUM(monthly_kpis[paying_users])
ARPPU = DIVIDE([Revenue], [Paying Users])
DAU = SUM(daily_kpis[dau])
DAU MAU = DIVIDE([DAU], [MAU])
Churn Rate = AVERAGE(powerbi_customer_features[churn_30d])
Avg Predicted LTV = AVERAGE(customer_segments[predicted_ltv])
High Risk Users = CALCULATE(DISTINCTCOUNT(customer_segments[user_id]), customer_segments[risk_segment] = "High Risk")
```

