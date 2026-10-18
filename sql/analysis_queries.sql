
-- 1. Calculate normal and attack traffic percentages
SELECT
    CASE
        WHEN label = 0 THEN 'Normal'
        ELSE 'Attack'
    END AS traffic_type,
    COUNT(*) AS total_records,
    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM network_traffic),
        2
    ) AS percentage
FROM network_traffic
GROUP BY label
ORDER BY label;


-- 2. Find the top 10 protocols by attack count
SELECT
    proto,
    COUNT(*) AS total_records,
    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS attack_records
FROM network_traffic
GROUP BY proto
ORDER BY attack_records DESC
LIMIT 10;


-- 3. Compare average source bytes by traffic type
SELECT
    CASE
        WHEN label = 0 THEN 'Normal'
        ELSE 'Attack'
    END AS traffic_type,
    ROUND(AVG(sbytes), 2) AS average_source_bytes,
    ROUND(AVG(dbytes), 2) AS average_destination_bytes
FROM network_traffic
GROUP BY label
ORDER BY label;


-- 4. Identify protocols with the highest attack percentage
SELECT
    proto,
    COUNT(*) AS total_records,
    ROUND(
        SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) * 100.0
        / COUNT(*),
        2
    ) AS attack_percentage
FROM network_traffic
GROUP BY proto
HAVING COUNT(*) >= 100
ORDER BY attack_percentage DESC
LIMIT 10;
