WITH exposure_funnel AS (
  SELECT 
    e.user_id,
    MIN(e.created_at) AS first_exposure_ts
  FROM `bigquery-public-data.thelook_ecommerce.events` AS e
  WHERE e.event_type = 'product'
    AND e.user_id IS NOT NULL
    AND e.created_at BETWEEN '2025-10-01' AND '2025-11-30'
  GROUP BY e.user_id
),

cohort_assignment AS (
  SELECT 
    ef.user_id,
    ef.first_exposure_ts,
    u.latitude AS user_lat,
    u.longitude AS user_long,
    CASE 
      WHEN MOD(ABS(FARM_FINGERPRINT(CONCAT(CAST(ef.user_id AS STRING), 'MSFT_EXP_2DAY_V2'))), 2) = 1 
      THEN 'Treatment' 
      ELSE 'Control' 
    END AS variant
  FROM exposure_funnel AS ef
  INNER JOIN `bigquery-public-data.thelook_ecommerce.users` AS u
    ON ef.user_id = u.id
  WHERE u.country = 'United States'
    AND u.latitude IS NOT NULL 
    AND u.longitude IS NOT NULL
),

pre_experiment_covariates AS (
  SELECT 
    ca.user_id,
    COALESCE(SUM(oi.sale_price), 0.0) AS pre_spend_covariate,
    COUNT(DISTINCT e.session_id) AS pre_sessions_covariate
  FROM cohort_assignment AS ca
  LEFT JOIN `bigquery-public-data.thelook_ecommerce.order_items` AS oi
    ON ca.user_id = oi.user_id
    AND oi.created_at BETWEEN '2025-08-01' AND '2025-09-30'
    AND oi.status NOT IN ('Cancelled')
  LEFT JOIN `bigquery-public-data.thelook_ecommerce.events` AS e
    ON ca.user_id = e.user_id
    AND e.created_at BETWEEN '2025-08-01' AND '2025-09-30'
  GROUP BY ca.user_id
),

post_exposure_activity AS (
  SELECT 
    ca.user_id,
    COUNT(DISTINCT oi.order_id) AS post_orders,
    COALESCE(SUM(oi.sale_price), 0.0) AS gross_revenue,
    COALESCE(SUM(p.cost), 0.0) AS cogs,
    COUNT(DISTINCT oi.id) AS total_shipped,
    COUNT(DISTINCT CASE WHEN oi.status = 'Returned' THEN oi.id END) AS total_returned,
    COUNT(DISTINCT CASE 
      WHEN oi.delivered_at IS NOT NULL 
       AND oi.shipped_at IS NOT NULL
       AND TIMESTAMP_DIFF(oi.delivered_at, oi.shipped_at, HOUR) >= 4
       AND TIMESTAMP_DIFF(oi.delivered_at, oi.created_at, HOUR) > 48 
      THEN oi.id 
    END) AS sla_breaches,
    COALESCE(SUM(4.50 + (ROUND(ST_DISTANCE(
      ST_GEOGPOINT(ca.user_long, ca.user_lat),
      ST_GEOGPOINT(dc.longitude, dc.latitude)
    )/1000.0, 2) * 0.0035)), 0.0) AS outbound_shipping_cost,
    COALESCE(SUM(CASE 
      WHEN oi.status = 'Returned' THEN 9.00 + (0.15 * oi.sale_price) 
      ELSE 0.0 
    END), 0.0) AS reverse_logistics_cost
  FROM cohort_assignment AS ca
  INNER JOIN `bigquery-public-data.thelook_ecommerce.order_items` AS oi
    ON ca.user_id = oi.user_id
    AND oi.created_at >= ca.first_exposure_ts
    AND oi.created_at <= '2025-11-30'
    AND oi.status NOT IN ('Cancelled')
  INNER JOIN `bigquery-public-data.thelook_ecommerce.products` AS p
    ON oi.product_id = p.id
  INNER JOIN `bigquery-public-data.thelook_ecommerce.inventory_items` AS inv
    ON oi.inventory_item_id = inv.id
  INNER JOIN `bigquery-public-data.thelook_ecommerce.distribution_centers` AS dc
    ON inv.product_distribution_center_id = dc.id
  GROUP BY ca.user_id
)

SELECT 
  ca.user_id,
  ca.variant,
  prec.pre_spend_covariate,
  prec.pre_sessions_covariate,
  CASE WHEN COALESCE(post.post_orders, 0) > 0 THEN 1 ELSE 0 END AS converted,
  COALESCE(post.post_orders, 0) AS post_orders,
  COALESCE(post.total_shipped, 0) AS total_shipped,
  COALESCE(post.total_returned, 0) AS total_returned,
  COALESCE(post.sla_breaches, 0) AS sla_breaches,
  COALESCE(post.gross_revenue, 0.0) AS gross_revenue,
  COALESCE(post.cogs, 0.0) AS cogs,
  COALESCE(post.outbound_shipping_cost, 0.0) AS outbound_shipping_cost,
  COALESCE(post.reverse_logistics_cost, 0.0) AS reverse_logistics_cost,
  ROUND(
    COALESCE(post.gross_revenue, 0.0) - 
    COALESCE(post.cogs, 0.0) - 
    COALESCE(post.outbound_shipping_cost, 0.0) - 
    COALESCE(post.reverse_logistics_cost, 0.0), 
    2
  ) AS net_contribution_margin
FROM cohort_assignment AS ca
INNER JOIN pre_experiment_covariates AS prec
  ON ca.user_id = prec.user_id
LEFT JOIN post_exposure_activity AS post
  ON ca.user_id = post.user_id;