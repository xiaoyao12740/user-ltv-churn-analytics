-- DAU
SELECT DATE(event_time) AS activity_date, COUNT(DISTINCT user_id) AS dau FROM events GROUP BY DATE(event_time) ORDER BY activity_date;
-- MAU and DAU/MAU
WITH d AS (SELECT DATE(event_time) dt, COUNT(DISTINCT user_id) dau FROM events GROUP BY DATE(event_time)), m AS (SELECT DATE_FORMAT(event_time,'%Y-%m-01') month_start, COUNT(DISTINCT user_id) mau FROM events GROUP BY DATE_FORMAT(event_time,'%Y-%m-01')) SELECT d.dt,d.dau,m.mau,d.dau/NULLIF(m.mau,0) dau_mau FROM d JOIN m ON DATE_FORMAT(d.dt,'%Y-%m-01')=m.month_start;
-- Revenue, payers, ARPU, ARPPU, AOV, paying rate
WITH a AS (SELECT DATE_FORMAT(event_time,'%Y-%m-01') month_start,COUNT(DISTINCT user_id) active_users FROM events GROUP BY 1), r AS (SELECT DATE_FORMAT(order_time,'%Y-%m-01') month_start,SUM(amount-refund_amount) revenue,COUNT(DISTINCT user_id) payers,COUNT(*) orders FROM transactions GROUP BY 1) SELECT a.month_start,r.revenue,r.revenue/NULLIF(a.active_users,0) arpu,r.revenue/NULLIF(r.payers,0) arppu,r.revenue/NULLIF(r.orders,0) aov,r.payers/NULLIF(a.active_users,0) paying_rate FROM a LEFT JOIN r USING(month_start);
-- Channel revenue
SELECT u.channel,SUM(t.amount-t.refund_amount) revenue,COUNT(DISTINCT t.user_id) paying_users FROM transactions t JOIN users u USING(user_id) GROUP BY u.channel ORDER BY revenue DESC;
-- Per-user order count and most recent activity
SELECT u.user_id,COUNT(DISTINCT t.order_id) order_count,MAX(e.event_time) last_active FROM users u LEFT JOIN transactions t USING(user_id) LEFT JOIN events e USING(user_id) GROUP BY u.user_id;

