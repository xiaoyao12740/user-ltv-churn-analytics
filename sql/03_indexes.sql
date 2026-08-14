CREATE INDEX idx_events_time_user ON events(event_time,user_id);
CREATE INDEX idx_events_user_time ON events(user_id,event_time);
CREATE INDEX idx_transactions_time_user ON transactions(order_time,user_id);
CREATE INDEX idx_transactions_user_time ON transactions(user_id,order_time);
CREATE INDEX idx_users_channel ON users(channel);
CREATE INDEX idx_predictions_risk_value ON customer_predictions(churn_probability,predicted_ltv);

