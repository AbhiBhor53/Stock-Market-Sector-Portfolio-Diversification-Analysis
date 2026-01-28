SELECT Sector, SUM(Revenuegrowth)/COUNT(Revenuegrowth) avg_growth
FROM Project7.DBO.sp500_companies
GROUP BY  Sector
ORDER BY  avg_growth DESC


SELECT Sector, SUM(Currentprice)/COUNT(Currentprice) avg_price
FROM Project7.DBO.sp500_companies
GROUP BY Sector
ORDER BY avg_price DESC

SELECT Sector, SUM(Marketcap)/COUNT(Marketcap) avg_CAP
FROM Project7.DBO.sp500_companies
GROUP BY Sector
ORDER BY avg_CAP DESC


SELECT Sector, SUM(Ebitda)/COUNT(Ebitda) avg_Ebitda
FROM Project7.DBO.sp500_companies
GROUP BY Sector
ORDER BY avg_Ebitda DESC

SELECT *
FROM Project7.DBO.sp500_companies c

select * from project7.dbo.sp500_index
select * from project7.dbo.sp500_stocks
