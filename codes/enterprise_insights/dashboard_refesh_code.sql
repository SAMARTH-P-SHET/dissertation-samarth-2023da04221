
--####################### EDLP MONITORING DASHBOARD - DATA REFRESH CODE - PART 1 - EDLP SUMMARY AND GC DETAILS ####################

DECLARE mxrdsdate DATE;
SET mxrdsdate= (SELECT CAST(MIN(a.cal_dt) AS DATE) FROM `schema.calendar_dim` a 
WHERE hg_yr_wk_nbr IN 
(SELECT b.hg_yr_wk_nbr FROM `schema.calendar_dim` b 
WHERE CAST(b.cal_dt AS DATE)=CAST(current_date AS DATE)));

CREATE OR REPLACE TABLE `schema.edlp_calendar` AS
SELECT hg_yr_wk_nbr, MIN(CAST(cal_dt AS DATE)) AS mn_date, MAX(CAST(cal_dt AS DATE)) AS mx_date,
ROW_NUMBER() OVER (ORDER BY hg_yr_wk_nbr - 52) AS WK_SEQ
FROM `schema.calendar_dim`
WHERE CAST(cal_dt AS DATE) BETWEEN DATE_SUB(mxrdsdate, INTERVAL 365 DAY) AND DATE_ADD(mxrdsdate, INTERVAL 110 DAY)
GROUP BY hg_yr_wk_nbr
ORDER BY 1;

CREATE OR REPLACE TABLE `schema.edlp_calendar_01` AS
SELECT MIN(hg_yr_wk_nbr) AS l52w, MAX(hg_yr_wk_nbr) AS processwk
FROM `schema.edlp_calendar`
WHERE wk_seq BETWEEN 1 AND 53;

SELECT processwk, l52w FROM `schema.edlp_calendar_01`;


-- 'product catlog' is an external table that has been created to pull the data from Hive.
DROP TABLE IF EXISTS schema.groupmap;
CREATE TABLE schema.groupmap
AS
SELECT 
  DISTINCT item_nbr, old_nbr, hg_iv_id, var_long_desc 
FROM schema.item_hierarchy_01_reg;


-- 'sales and inventory table' is an external table that has been created to pull the data from Hive.
DROP TABLE IF EXISTS schema.weekdt;
CREATE TABLE schema.weekdt AS
SELECT WKDT, ROW_NUMBER() OVER (ORDER by WKDT DESC) SQ FROM (
SELECT 
  DISTINCT hg_yr_wk_nbr WKDT 
FROM schema.prcg_sales_invt_full_wkly 
WHERE hg_yr_wk_nbr>202200
) X;


-- 'pmt_fchw_store' is an external table that has been created to pull the data from Hive.
DROP TABLE IF EXISTS schema.item_hier_v;
CREATE TABLE schema.item_hier_v AS
SELECT 
  a.hg_yr_wk_nbr hg_yr_wk, d.hg_region, a.item_nbr, a.upc_nbr, b.hg_iv_id, b.var_desc,
  b.sbu_nbr, b.sbu_desc, 
  b.dept_nbr, b.dept_desc,
  b.dept_category_nbr, b.dept_category_desc,
  CASE WHEN b.brand_type IS NULL THEN 'NB' ELSE b.brand_type END brand_type,
  SUM(a.wkly_sales_amt) hg_sales, 
  SUM(a.wkly_qty) hg_units
FROM schema.prcg_sales_invt_full_wkly a
INNER JOIN 
(SELECT 
  DISTINCT sbu_nbr, sbu_desc, dept_nbr, dept_desc,dept_category_nbr, dept_category_desc,brand_type, item_nbr, hg_iv_id, var_long_desc var_desc 
 FROM schema.item_hierarchy_01_reg) b
ON a.item_nbr = b.item_nbr
INNER JOIN
(SELECT 
  MAX(wkdt)-200 wk 
 FROM schema.weekdt) c
ON a.hg_yr_wk_nbr >= wk
LEFT JOIN 
(SELECT 
  DISTINCT store_nbr, hg_region 
 FROM schema.pmt_fchw_store) d
ON a.store_nbr = d.store_nbr
GROUP BY a.hg_yr_wk_nbr, d.hg_region, a.item_nbr, a.upc_nbr, b.hg_iv_id, b.var_desc, b.sbu_nbr, b.sbu_desc, b.dept_nbr, b.dept_desc,
b.dept_category_nbr, b.dept_category_desc, b.brand_type;


DROP TABLE IF EXISTS schema.item_hier_vol;
CREATE TABLE schema.item_hier_vol AS
SELECT 
  hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc, upc_nbr, hg_iv_id, var_desc, brand_type, 
  SUM(hg_sales) hg_sales, SUM(hg_units) hg_units
FROM schema.item_hier_v  
GROUP BY hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc, upc_nbr, hg_iv_id, var_desc, brand_type;


DROP TABLE IF EXISTS schema.item_hier_vol2;
CREATE TABLE schema.item_hier_vol2 AS
SELECT 
  DISTINCT hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc, upc_nbr, hg_iv_id,var_desc gc_description, 
  brand_type, hg_sales, hg_units 
  FROM schema.item_hier_vol 
UNION DISTINCT
SELECT 
  hg_yr_wk,'NATIONAL' hg_region, sbu_desc, dept_desc, dept_category_desc, upc_nbr, hg_iv_id, var_desc gc_description, brand_type,
  SUM(hg_sales) hg_sales, SUM(hg_units) hg_units 
FROM schema.item_hier_vol
GROUP BY hg_yr_wk, sbu_desc, dept_desc, dept_category_desc, upc_nbr, hg_iv_id, var_desc, brand_type;


-- 'Store price weekly data' is an external table that has been created to pull the data from Hive.
DROP TABLE IF EXISTS schema.hg_price1 ;
CREATE TABLE IF NOT EXISTS schema.hg_price1
AS
SELECT 
  DISTINCT hg_wk_nbr+190000 hg_week, a.item_nbr, store_nbr, region_nm region, live_price_amt hg_live, defined_reg_price_amt hg_reg,
  b.hg_iv_id group_code
FROM schema.hg_fchw_store_price_wkly a
LEFT JOIN 
(SELECT 
  DISTINCT item_nbr, hg_iv_id 
 FROM schema.groupmap) b 
ON a.item_nbr = b.item_nbr 
WHERE hg_wk_nbr+190000>(SELECT wkdt FROM  schema.weekdt WHERE sq = 52);


DROP TABLE IF EXISTS schema.hg_price2;
CREATE TABLE IF NOT EXISTS schema.hg_price2 AS
SELECT *  FROM
(
SELECT 
  hg_week,'NATIONAL' hg_region, group_code, hg_reg, hg_live,
  RANK() OVER (PARTITION BY hg_week, group_code ORDER BY COUNT(*) DESC, hg_reg, hg_live) AS seq
FROM schema.hg_price1  
GROUP BY hg_week, group_code, hg_reg, hg_live
UNION ALL
SELECT 
  hg_week, region, group_code , hg_reg, hg_live,
  RANK() OVER (PARTITION BY hg_week, region, group_code ORDER BY COUNT(*) DESC, hg_reg, hg_live) AS seq
FROM schema.hg_price1  
GROUP BY hg_week, region, group_code, hg_reg, hg_live
) X WHERE seq = 1;


-- 'instore weekly data' is an external table that has been created to pull the data from Hive.
DROP TABLE IF EXISTS schema.comp_price1;
CREATE TABLE schema.comp_price1 AS
SELECT 
  DISTINCT hg_week+190000 hg_week, hg_store_id, hg_region, upc_nbr, item_nbr, banner_name, 
  CASE WHEN equalization_factor>0 THEN  comp_reg*equalization_factor ELSE comp_reg END comp_reg_eq, 
  CASE WHEN equalization_factor>0 THEN  comp_live*equalization_factor ELSE comp_live END comp_live_eq,
  b.hg_iv_id  group_code
FROM schema.rds_walcaninstoreweekly
 a
LEFT JOIN 
(SELECT 
  DISTINCT old_nbr, hg_iv_id 
 FROM schema.groupmap) b 
ON a.item_nbr = b.old_nbr 
WHERE hg_week+190000>(SELECT wkdt FROM schema.weekdt WHERE sq = 52);


DROP TABLE IF EXISTS schema.comp_price2;
CREATE TABLE IF NOT EXISTS schema.comp_price2 
AS
SELECT 
  hg_week, hg_store_id, hg_region, upc_nbr, item_nbr, group_code, 'MKT_LOW' banner_name,
  MIN(CASE WHEN comp_reg_eq IS NOT NULL THEN comp_reg_eq ELSE comp_live_eq END)  comp_reg, 
  MIN(CASE WHEN comp_live_eq IS NOT NULL THEN comp_live_eq ELSE comp_reg_eq END ) comp_live
FROM schema.comp_price1
WHERE  comp_reg_eq>0.1 OR comp_live_eq>0.1
GROUP BY hg_week, hg_store_id, hg_region, upc_nbr, item_nbr, group_code;


DROP TABLE IF EXISTS schema.comp_price3;
CREATE TABLE IF NOT EXISTS schema.comp_price3 
AS
SELECT *  FROM
(
SELECT 
  hg_week,'NATIONAL' hg_region, group_code, comp_reg, comp_live,
  RANK() OVER (PARTITION BY hg_week, group_code ORDER BY COUNT(*) DESC, comp_reg, comp_live ) AS seq
FROM schema.comp_price2   
GROUP BY hg_week, group_code, comp_reg, comp_live
UNION ALL
SELECT 
  hg_week, hg_region, group_code , comp_reg, comp_live,
  RANK() OVER (PARTITION BY hg_week,hg_region,group_code ORDER BY COUNT(*) DESC, comp_reg, comp_live) AS seq
FROM schema.comp_price2  
GROUP BY hg_week, hg_region, group_code, comp_reg, comp_live
) X WHERE seq = 1;


-- SOURCE TABLE: hgt.dashboard_source

SELECT Max(PROMO_YR_WK) FROM hgt-camerch-pricing-prod.COMP_TABLES.dashboard;


 DROP TABLE IF EXISTS schema.dashboard1;
CREATE TABLE schema.dashboard1 AS
SELECT
  promo_yr_wk, region, market_description, period_description,
  regexp_replace(CAST(REPLACE(LTRIM(REPLACE(UPC, '0', ' ')), ' ', '0') AS STRING), 'BP[0-9]','') niq_upc,
  SUM(safe_cast(`_` AS FLOAT64)) sales, 
  SUM(safe_cast(`__ya` AS FLOAT64)) sales_ya,
  SUM(safe_cast(units AS FLOAT64)) units, 
  SUM(safe_cast(units_ya AS FLOAT64)) units_ya,
  SUM(safe_cast(`any_promo__` AS FLOAT64)) promo_sales, 
  SUM(safe_cast(`any_promo___ya` AS FLOAT64)) promo_sales_ya,
  SUM(safe_cast(any_promo_units AS FLOAT64)) promo_units, 
  SUM(safe_cast(any_promo_units_ya AS FLOAT64)) promo_units_ya
FROM  hgt.dashboard_source
WHERE market_description NOT LIKE '%CALGARY%' AND market_description NOT LIKE '%TORONTO%' AND market_description NOT LIKE '%VANCOUVER%' AND market_description NOT LIKE '%CENSUS%' AND market_description NOT LIKE '%DISCOUNT%' AND market_description NOT LIKE '%MARITIMES%'
AND period_description IS NOT NULL
GROUP BY promo_yr_wk, region, market_description, period_description, 
  regexp_replace(CAST(REPLACE(LTRIM(REPLACE(UPC, '0', ' ')), ' ', '0') AS STRING), 'BP[0-9]','');


DROP TABLE IF EXISTS schema.dashboard2;
CREATE TABLE schema.dashboard2 AS
SELECT 
  a.*,  b.hg_upc
FROM schema.dashboard1 a
LEFT JOIN schema.niq_map_1 b
ON upper(a.region) = Upper(b.region) 
AND a.niq_upc = b.niq_upc;


DROP TABLE IF EXISTS schema.dashboard3;
CREATE TABLE schema.dashboard3 AS
SELECT 
  A.*, 
  CASE WHEN c.w_c IS NULL THEN 0 ELSE c.w_c END hg_dup
FROM schema.dashboard2 A
LEFT JOIN
(SELECT 
  region, niq_upc, COUNT(DISTINCT hg_upc) w_c
FROM schema.dashboard2 
GROUP BY region, niq_upc)c
ON upper(a.region) = upper(c.region) 
AND a.niq_upc = c.niq_upc;


DROP TABLE IF EXISTS schema.dashboard4;
CREATE TABLE schema.dashboard4 AS
SELECT
  promo_yr_wk, region, market_description, period_description, niq_upc, hg_upc,	
  CASE WHEN hg_dup>1 THEN sales/hg_dup ELSE sales END sales,
  CASE WHEN hg_dup>1 THEN sales_ya/hg_duP ELSE sales_ya END sales_ya,
  CASE WHEN hg_dup>1 THEN units/hg_dup ELSE units END units,
  CASE WHEN hg_dup>1 THEN units_ya/hg_dup ELSE units_ya END units_ya,
  CASE WHEN hg_dup>1 THEN promo_sales/hg_dup ELSE promo_sales END promo_sales,
  CASE WHEN hg_dup>1 THEN promo_sales_ya/hg_dup ELSE promo_sales_ya END promo_sales_ya,
  CASE WHEN hg_dup>1 THEN promo_units/hg_dup ELSE promo_units END promo_units,
  CASE WHEN hg_dup>1 THEN promo_units_ya/hg_dup ELSE promo_units_ya END promo_units_ya
FROM schema.dashboard3
WHERE sales > 0 OR sales_ya >0;


DROP TABLE IF EXISTS schema.dashboard5;
CREATE TABLE schema.dashboard5 AS
SELECT
  promo_yr_wk, region, market_description, period_description, hg_upc,
  SUM(sales) sales,	
  SUM(sales_ya) sales_ya,	
  SUM(units) units,	
  SUM(units_ya) units_ya,	
  SUM(promo_sales) promo_sales,	
  SUM(promo_sales_ya) promo_sales_ya,	
  SUM(promo_units) promo_units,	
  SUM(promo_units_ya) promo_units_ya
FROM schema.dashboard4
GROUP BY promo_yr_wk, region, market_description, period_description, hg_upc
UNION DISTINCT
SELECT 
  promo_yr_wk, 'NATIONAL' region, market_description, period_description, hg_upc,
  SUM(sales) sales,	
  SUM(sales_ya) sales_ya,	
  SUM(units) units,	
  SUM(units_ya) units_ya,	
  SUM(promo_sales) promo_sales,	
  SUM(promo_sales_ya) promo_sales_ya,	
  SUM(promo_units) promo_units,	
  SUM(promo_units_ya) promo_units_ya
FROM schema.dashboard4
GROUP BY promo_yr_wk, market_description, period_description, hg_upc;


DROP TABLE IF EXISTS schema.dashboard6;
CREATE TABLE schema.dashboard6 AS
SELECT 
  PROMO_YR_WK,	REGION,	PERIOD_DESCRIPTION,	hg_UPC,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN sales			     end) total_mkt_sales,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN sales_ya		   end) total_mkt_sales_ya,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN units			     end) total_mkt_units,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN units_ya		   end) total_mkt_units_ya,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN promo_sales	   end) total_mkt_promo_sales,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN promo_sales_ya end) total_mkt_promo_sales_ya,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN promo_units	   end) total_mkt_promo_units,
  SUM(CASE WHEN market_description LIKE '%ALL CHANNELS%' THEN promo_units_ya end) total_mkt_promo_units_ya,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN sales			     end) gdm_sales,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN sales_ya		   end) gdm_sales_ya,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN units			     end) gdm_units,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN units_ya		   end) gdm_units_ya,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN promo_sales	   end) gdm_promo_sales,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN promo_sales_ya end) gdm_promo_sales_ya,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN promo_units	   end) gdm_promo_units,
  SUM(CASE WHEN market_description LIKE '%GB +DR +MM%'   THEN promo_units_ya end) gdm_promo_units_ya,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN sales			     end) HG_MART_sales,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN sales_ya		   end) HG_MART_sales_ya,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN units			     end) HG_MART_units,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN units_ya		   end) HG_MART_units_ya,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN promo_sales	   end) HG_MART_promo_sales,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN promo_sales_ya end) HG_MART_promo_sales_ya,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN promo_units	   end) HG_MART_promo_units,
  SUM(CASE WHEN market_description LIKE '%HG_MART%'      THEN promo_units_ya end) HG_MART_promo_units_ya
FROM schema.dashboard5
GROUP BY promo_yr_wk, region, period_description, hg_upc;


DROP TABLE IF EXISTS schema.dashboard7;
CREATE TABLE schema.dashboard7 AS
SELECT 
  PROMO_YR_WK, REGION, hg_UPC,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_sales END)total_mkt_sales_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_sales_ya END)total_mkt_sales_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_units END)total_mkt_units_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_units_ya END)total_mkt_units_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_promo_sales END)total_mkt_promo_sales_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_promo_sales_ya END)total_mkt_promo_sales_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_promo_units END)total_mkt_promo_units_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN total_mkt_promo_units_ya END)total_mkt_promo_units_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_sales END)gdm_sales_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_sales_ya END)gdm_sales_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_units END)gdm_units_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_units_ya END)gdm_units_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_promo_sales END)gdm_promo_sales_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_promo_sales_ya END)gdm_promo_sales_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_promo_units END)gdm_promo_units_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN gdm_promo_units_ya END)gdm_promo_units_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_sales END)HG_MART_sales_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_sales_ya END)HG_MART_sales_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_units END)HG_MART_units_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_units_ya END)HG_MART_units_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_promo_sales END)HG_MART_promo_sales_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_promo_sales_ya END)HG_MART_promo_sales_ya_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_promo_units END)HG_MART_promo_units_1w,
  SUM(CASE WHEN period_description LIKE '%1 w/e%' THEN HG_MART_promo_units_ya END)HG_MART_promo_units_ya_1w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_sales END)total_mkt_sales_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_sales_ya END)total_mkt_sales_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_units END)total_mkt_units_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_units_ya END)total_mkt_units_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_promo_sales END)total_mkt_promo_sales_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_promo_sales_ya END)total_mkt_promo_sales_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_promo_units END)total_mkt_promo_units_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN total_mkt_promo_units_ya END)total_mkt_promo_units_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_sales END)gdm_sales_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_sales_ya END)gdm_sales_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_units END)gdm_units_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_units_ya END)gdm_units_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_promo_sales END)gdm_promo_sales_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_promo_sales_ya END)gdm_promo_sales_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_promo_units END)gdm_promo_units_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN gdm_promo_units_ya END)gdm_promo_units_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_sales END)HG_MART_sales_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_sales_ya END)HG_MART_sales_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_units END)HG_MART_units_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_units_ya END)HG_MART_units_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_promo_sales END)HG_MART_promo_sales_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_promo_sales_ya END)HG_MART_promo_sales_ya_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_promo_units END)HG_MART_promo_units_4w,
  SUM(CASE WHEN period_description LIKE '%4 w/e%' THEN HG_MART_promo_units_ya END)HG_MART_promo_units_ya_4w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_sales END)total_mkt_sales_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_sales_ya END)total_mkt_sales_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_units END)total_mkt_units_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_units_ya END)total_mkt_units_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_promo_sales END)total_mkt_promo_sales_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_promo_sales_ya END)total_mkt_promo_sales_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_promo_units END)total_mkt_promo_units_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN total_mkt_promo_units_ya END)total_mkt_promo_units_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_sales END)gdm_sales_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_sales_ya END)gdm_sales_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_units END)gdm_units_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_units_ya END)gdm_units_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_promo_sales END)gdm_promo_sales_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_promo_sales_ya END)gdm_promo_sales_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_promo_units END)gdm_promo_units_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN gdm_promo_units_ya END)gdm_promo_units_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_sales END)HG_MART_sales_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_sales_ya END)HG_MART_sales_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_units END)HG_MART_units_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_units_ya END)HG_MART_units_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_promo_sales END)HG_MART_promo_sales_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_promo_sales_ya END)HG_MART_promo_sales_ya_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_promo_units END)HG_MART_promo_units_12w,
  SUM(CASE WHEN period_description LIKE '%12 w/e%' THEN HG_MART_promo_units_ya END)HG_MART_promo_units_ya_12w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_sales END)total_mkt_sales_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_sales_ya END)total_mkt_sales_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_units END)total_mkt_units_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_units_ya END)total_mkt_units_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_promo_sales END)total_mkt_promo_sales_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_promo_sales_ya END)total_mkt_promo_sales_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_promo_units END)total_mkt_promo_units_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN total_mkt_promo_units_ya END)total_mkt_promo_units_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_sales END)gdm_sales_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_sales_ya END)gdm_sales_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_units END)gdm_units_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_units_ya END)gdm_units_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_promo_sales END)gdm_promo_sales_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_promo_sales_ya END)gdm_promo_sales_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_promo_units END)gdm_promo_units_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN gdm_promo_units_ya END)gdm_promo_units_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_sales END)HG_MART_sales_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_sales_ya END)HG_MART_sales_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_units END)HG_MART_units_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_units_ya END)HG_MART_units_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_promo_sales END)HG_MART_promo_sales_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_promo_sales_ya END)HG_MART_promo_sales_ya_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_promo_units END)HG_MART_promo_units_52w,
  SUM(CASE WHEN period_description LIKE '%52 w/e%' THEN HG_MART_promo_units_ya END)HG_MART_promo_units_ya_52w
FROM schema.dashboard6
GROUP BY promo_yr_wk, region, hg_upc;


DROP TABLE IF EXISTS schema.dashboard8;
CREATE TABLE schema.dashboard8 AS
SELECT  
  a.hg_yr_wk, a.hg_region, a.sbu_desc, a.dept_desc, a.dept_category_desc, a.upc_nbr, a.hg_iv_id, a.gc_description, a.brand_type, 
  b.total_mkt_sales_1w, b.total_mkt_units_1w, b.total_mkt_promo_sales_1w, b.total_mkt_promo_units_1w, b.gdm_sales_1w, b.gdm_units_1w,
  b.gdm_promo_sales_1w, b.gdm_promo_units_1w, b.HG_MART_sales_1w, b.HG_MART_units_1w, b.HG_MART_promo_sales_1w, b.HG_MART_promo_units_1w,
  b.total_mkt_units_52w, CAST(a.hg_sales AS FLOAT64) hg_sales, CAST(a.hg_units AS FLOAT64) hg_units
FROM schema.item_hier_vol2 A
LEFT JOIN schema.dashboard7 B
ON a.hg_yr_wk = (b.promo_yr_wk+190000) 
AND  UPPER(a.hg_region) = UPPER(b.region) 
AND CAST(a.upc_nbr AS INT64) = CAST(b.hg_upc AS INT64)
WHERE a.hg_region IS NOT NULL;


DROP TABLE IF EXISTS schema.dashboard9;
CREATE TABLE schema.dashboard9 AS
SELECT 
  hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc,  hg_iv_id, gc_description,brand_type,
  SUM(total_mkt_sales_1w) total_mkt_sales_1w, 
  SUM(total_mkt_units_1w) total_mkt_units_1w,
  SUM(total_mkt_promo_sales_1w) total_mkt_promo_sales_1w, 
  SUM(total_mkt_promo_units_1w) total_mkt_promo_units_1w,
  SUM(gdm_sales_1w) gdm_sales_1w, 
  SUM(gdm_units_1w) gdm_units_1w,
  SUM(gdm_promo_sales_1w) gdm_promo_sales_1w, 
  SUM(gdm_promo_units_1w) gdm_promo_units_1w,
  SUM(HG_MART_sales_1w) HG_MART_sales_1w, 
  SUM(HG_MART_units_1w) HG_MART_units_1w,
  SUM(HG_MART_promo_sales_1w) HG_MART_promo_sales_1w, 
  SUM(HG_MART_promo_units_1w) HG_MART_promo_units_1w,
  SUM(total_mkt_units_52w) total_mkt_units_52w, 
  SUM(hg_sales) hg_sales, 
  SUM(hg_units) hg_units
FROM schema.dashboard8
GROUP BY hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc, hg_iv_id, gc_description, brand_type;


DROP TABLE IF EXISTS schema.dashboard10;
CREATE TABLE schema.dashboard10 AS
SELECT 
  DISTINCT a.hg_yr_wk, a.hg_region, a.sbu_desc, a.dept_desc, a.dept_category_desc, a.hg_iv_id, a.gc_description, a.brand_type, 
  a.total_mkt_sales_1w, a.total_mkt_units_1w, a.total_mkt_promo_sales_1w, a.total_mkt_promo_units_1w, a.gdm_sales_1w, a.gdm_units_1w,
  a.gdm_promo_sales_1w, a.gdm_promo_units_1w, a.HG_MART_sales_1w, a.HG_MART_units_1w, a.HG_MART_promo_sales_1w, a.HG_MART_promo_units_1w,
  a.total_mkt_units_52w, a.hg_sales, a.hg_units, b.hg_live, b.hg_reg, c.comp_live, c.comp_reg
FROM schema.dashboard9 A
LEFT JOIN schema.hg_price2 B
ON a.hg_yr_wk = b.hg_week 
AND UPPER(a.hg_region) = UPPER(b.hg_region) AND a.hg_iv_id = b.group_code
LEFT JOIN schema.comp_price3 c
ON a.hg_yr_wk = c.hg_week 
AND UPPER(a.hg_region) = UPPER(c.hg_region) 
AND a.hg_iv_id = c.group_code;


DROP TABLE IF EXISTS schema.dashboard11 ;
CREATE TABLE schema.dashboard11 AS
SELECT 
  DISTINCT hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc, hg_iv_id, gc_description, brand_type, total_mkt_sales_1w,
  total_mkt_units_1w,total_mkt_promo_sales_1w,total_mkt_promo_units_1w, gdm_sales_1w, gdm_units_1w, gdm_promo_sales_1w, 
  gdm_promo_units_1w, HG_MART_sales_1w, HG_MART_units_1w, HG_MART_promo_sales_1w, HG_MART_promo_units_1w,
  CASE WHEN total_mkt_units_52w IS NULL OR total_mkt_units_52w<1 THEN 1 ELSE total_mkt_units_52w END total_mkt_units_52w,
  hg_sales, hg_units,hg_live, hg_reg, 
  ROUND(CASE WHEN comp_live IS NULL OR comp_live = 0 THEN (gdm_promo_sales_1w - HG_MART_promo_sales_1w)/NULLIF((gdm_promo_units_1w-HG_MART_promo_units_1w),0) ELSE COMP_LIVE END,2) comp_live,
  ROUND(CASE WHEN comp_reg IS NULL OR comp_reg = 0 THEN (total_mkt_sales_1w - HG_MART_sales_1w)/NULLIF((total_mkt_units_1w-HG_MART_units_1w),0) ELSE comp_reg END,2) comp_reg
FROM schema.dashboard10;


DROP TABLE IF EXISTS schema.dashboard11a ;
CREATE TABLE schema.dashboard11a AS
SELECT 
  DISTINCT hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc, hg_iv_id, gc_description, brand_type, total_mkt_sales_1w,
  total_mkt_units_1w,total_mkt_promo_sales_1w,total_mkt_promo_units_1w, gdm_sales_1w,gdm_units_1w,gdm_promo_sales_1w,gdm_promo_units_1w,
  HG_MART_sales_1w,HG_MART_units_1w, HG_MART_promo_sales_1w,HG_MART_promo_units_1w,total_mkt_units_52w,hg_sales, hg_units, hg_live, hg_reg, 
  CASE WHEN (comp_live IS NULL OR comp_live<0) AND (comp_reg>0 AND comp_reg IS NOT NULL) THEN comp_reg  ELSE comp_live END AS comp_live,
  CASE WHEN comp_reg>0 AND comp_reg IS NOT NULL THEN comp_reg END AS comp_reg
FROM schema.dashboard11;


DROP TABLE IF EXISTS schema.dashboard12;
CREATE TABLE schema.dashboard12 AS
SELECT 
  DISTINCT hg_yr_wk, hg_region, sbu_desc, dept_desc, dept_category_desc, hg_iv_id, gc_description, brand_type, total_mkt_sales_1w,
  total_mkt_units_1w,total_mkt_promo_sales_1w,total_mkt_promo_units_1w, gdm_sales_1w,gdm_units_1w,gdm_promo_sales_1w,gdm_promo_units_1w,
  HG_MART_sales_1w,HG_MART_units_1w, HG_MART_promo_sales_1w,HG_MART_promo_units_1w,total_mkt_units_52w,hg_sales, hg_units, hg_live, hg_reg, 
  CASE WHEN comp_live IS NULL THEN comp_reg END AS comp_live, comp_reg,
  CASE WHEN comp_live> 0 AND hg_live> 0  THEN total_mkt_units_52w*hg_live END hg_live_dlr,
  CASE WHEN comp_reg>0 AND hg_reg> 0  THEN total_mkt_units_52w*hg_reg END hg_reg_dlr,
  CASE WHEN comp_live > 0 AND hg_live>0  THEN  total_mkt_units_52w*comp_reg END comp_live_dlr,
  CASE WHEN comp_reg>0 AND hg_reg>0  THEN total_mkt_units_52w*comp_reg END comp_reg_dlr
FROM schema.dashboard11a;


--SOURCE TABLE: hgt-camerch-pricing-prod.PRICING_HUB_DATA_VM.WKLY_REGION_GC_SALES_METRICS_ACTUALS
DROP TABLE IF EXISTS schema.hg_p;
CREATE TABLE schema.hg_p AS
SELECT 
  promo_hg_yr_wk+190000 promo_hg_yr_wk, region, group_code_nbr,
  SUM(total_units) units,
  SUM(total_sales) sales,
  SUM(CASE WHEN (on_flyer = TRUE OR on_cpp = TRUE OR price_mechanic <> 'EDLP') THEN total_units END ) promo_units,
  SUM(CASE WHEN (on_flyer = TRUE OR on_cpp = TRUE OR price_mechanic <> 'EDLP') THEN total_sales END ) promo_sales,
  SUM(base_units) baseline_units,
  SUM(total_units - base_units) incremental_units
FROM hgt.schema.WKLY_REGION_GC_SALES_METRICS_ACTUALS
WHERE promo_hg_yr_wk+190000 BETWEEN (SELECT wkdt FROM schema.weekdt WHERE sq = 52) 
AND (SELECT wkdt FROM schema.weekdt WHERE sq = 1)
GROUP BY promo_hg_yr_wk+190000 ,region, group_code_nbr
UNION DISTINCT
SELECT promo_hg_yr_wk+190000 promo_hg_yr_wk,'national' region, group_code_nbr,
SUM(total_units) units,
SUM(total_sales) sales,
SUM(CASE WHEN (on_flyer = TRUE OR on_cpp = TRUE OR price_mechanic <> 'EDLP') THEN total_units END ) promo_units,
SUM(CASE WHEN (on_flyer = TRUE OR on_cpp = TRUE OR price_mechanic <> 'EDLP') THEN total_sales END ) promo_sales,
SUM(base_units) baseline_units,
SUM(total_units - base_units) incremental_units
FROM hgt-camerch-pricing-prod.PRICING_HUB_DATA_VM.WKLY_REGION_GC_SALES_METRICS_ACTUALS
WHERE promo_hg_yr_wk+190000 BETWEEN (SELECT wkdt FROM schema.weekdt WHERE sq = 52) 
AND (SELECT wkdt FROM schema.weekdt WHERE sq = 1)
GROUP BY promo_hg_yr_wk+190000, group_code_nbr;


DROP TABLE IF EXISTS schema.dashboard13;
CREATE TABLE schema.dashboard13 AS
SELECT 
  DISTINCT a.hg_yr_wk, a.hg_region, a.sbu_desc, a.dept_desc, a.dept_category_desc, a.hg_iv_id, a.gc_description, a.brand_type, 
  a.total_mkt_sales_1w,a.total_mkt_units_1w,a.total_mkt_promo_sales_1w,a.total_mkt_promo_units_1w, a.gdm_sales_1w,a.gdm_units_1w,
  a.gdm_promo_sales_1w,a.gdm_promo_units_1w,'' cat_sales, a.HG_MART_sales_1w,a.HG_MART_units_1w, a.HG_MART_promo_sales_1w,
  a.HG_MART_promo_units_1w,a.total_mkt_units_52w, '' edlp,a.hg_live, a.hg_reg, a.comp_live, a.comp_reg, a.hg_live_dlr, a.hg_reg_dlr, 
  a.comp_live_dlr, a.comp_reg_dlr,
  CASE WHEN b.sales IS NULL THEN a.hg_sales ELSE b.sales END hg_sales, 
  CASE WHEN b.units IS NULL THEN a.hg_units ELSE b.units END hg_units,
  b.promo_units, b.promo_sales, b.baseline_units, b.incremental_units
FROM schema.dashboard12 a
LEFT JOIN schema.hg_p b
ON a.hg_yr_wk = b.promo_hg_yr_wk 
AND a.hg_iv_id = b.group_code_nbr 
AND UPPER(a.hg_region) = UPPER(b.region);


CREATE OR REPLACE EXTERNAL TABLE schema.edlp_actioned
OPTIONS (
  format = 'CSV',
  uris = ['gs://01f9b3d6809911038d08ab2a4/samarth/001_EDLP_Actioned/EDLP_Actioned.csv'],
  skip_leading_rows = 1,
  autodetect = TRUE
);


DROP TABLE schema.dashboard_final1;
CREATE TABLE schema.dashboard_final1 AS
SELECT 
CONCAT('1', SUBSTRING(CAST(hg_yr_wk AS STRING), 3, 2), SUBSTRING(CAST(hg_yr_wk AS STRING), 5, 2)) as `hg Week`,
hg_region AS `Region`,
sbu_desc AS `SBU`,
dept_desc AS `Department`,
dept_category_desc AS `Category`,
hg_iv_id AS `Group Code`,
gc_description AS `Group Code Description`,
brand_type AS `Brand Type`,
total_mkt_sales_1w AS `Total Mkt Sales 1W`,
total_mkt_units_1w AS `Total Mkt Units 1W`,
total_mkt_promo_sales_1w AS `Total Mkt Promo Sales 1W`,
total_mkt_promo_units_1w AS `Total Mkt Promo Units 1W`,
gdm_sales_1w AS `GDM Sales 1W`,
gdm_units_1w AS `GDM Units 1W`,
gdm_promo_sales_1w AS `GDM Promo Sales 1W`,
gdm_promo_units_1w AS `GDM Promo Units 1W`,
cat_sales AS `Cat Sales`,
HG_MART_sales_1w AS `HG_MART Sales 1W`,
HG_MART_units_1w AS `HG_MART Units 1W`,
HG_MART_promo_sales_1w AS `HG_MART Promo Sales 1W`,
HG_MART_promo_units_1w AS `HG_MART Promo Units 1W`,
total_mkt_units_52w AS `Total Mkt Units 52W`,
CASE WHEN b.`Group Code` IS NOT NULL THEN 1 ELSE 0 END AS `EDLP`,
hg_live AS `hg Live`,
hg_reg AS `hg Reg`,
comp_live AS `Comp Live`,
comp_reg AS `Comp Reg`,
hg_live_dlr AS `hg Live Dol`,
hg_reg_dlr AS `hg Reg Dol`,
comp_live_dlr AS `Comp Live Dol`,
comp_reg_dlr AS `Comp Reg Dol`,
hg_sales AS `hg T Sales`,
hg_units AS `hg T Units`,
promo_units AS `hg P Units`,
promo_sales AS `hg P Sales`,
baseline_units AS `hg_Baseline`,
incremental_units AS `hg_Incremental`
FROM schema.dashboard13 a
LEFT JOIN schema.edlp_actioned_23062025 b
ON CAST(a.hg_iv_id AS string) = CAST(b.`Group Code` AS string)
WHERE b.`Effective date` < CURRENT_DATE() 
OR b.`Group Code` IS NULL
AND CAST(a.hg_yr_wk AS INT64) BETWEEN (SELECT l52w FROM `schema.edlp_calendar_01`) and (SELECT processwk FROM `schema.edlp_calendar_01`);
   
   
DROP TABLE schema.dashboard_1;
CREATE TABLE schema.dashboard_1 AS
SELECT distinct * FROM schema.dashboard_final1 
WHERE `hg Week` BETWEEN '12421' AND '12520';


CREATE OR REPLACE TABLE schema.niq_master_filter AS
SELECT 
  DISTINCT Region, SBU,	Department,	Category, `Group Code`, `Group Code Description`, `Brand Type`, `EDLP`,
  CASE WHEN EDLP = 1 THEN 'YES' ELSE 'NO' END AS EDLP_Actioned
FROM  schema.dashboard_1
ORDER BY 1,2,3,4;
