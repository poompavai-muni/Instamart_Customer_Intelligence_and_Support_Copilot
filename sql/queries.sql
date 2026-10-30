-- Q1_where_having
SELECT customer_id, SUM(order_amount) AS high_value_spend
FROM orders WHERE status='completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000
ORDER BY high_value_spend DESC;

-- Q2_city_category_revenue
SELECT c.city, p.category, ROUND(SUM(o.order_amount),3) AS revenue
FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id)
WHERE o.status='completed' GROUP BY c.city, p.category ORDER BY c.city, p.category;

-- Q3_products_above_average
SELECT product_id, product_name, price_inr FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4_customers_no_completed_order
SELECT c.customer_id FROM customers c
LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5_spend_above_average
WITH customer_spend AS (
 SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
 WHERE status='completed' GROUP BY customer_id
), average_spend AS (SELECT AVG(total_spend) AS avg_spend FROM customer_spend)
SELECT customer_id, total_spend, avg_spend FROM customer_spend CROSS JOIN average_spend
WHERE total_spend > avg_spend ORDER BY total_spend DESC;

-- Q6_city_spend_ranks
WITH customer_spend AS (
 SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
 FROM customers c JOIN orders o USING(customer_id)
 WHERE o.status='completed' GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
 RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_num
FROM customer_spend ORDER BY city, spend_rank, customer_id;

-- Q7_daily_running_revenue
WITH daily AS (
 SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)
)
SELECT day, ROUND(revenue,3) AS revenue,
 ROUND(SUM(revenue) OVER(ORDER BY day),3) AS cumulative_revenue
FROM daily ORDER BY day;

