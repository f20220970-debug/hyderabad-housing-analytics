# Results summary

- Data: 2,464 Hyderabad listings after cleaning (552 resale), 238 localities; median price INR 77 lakh, median INR 5,000 per sq ft.
- Price prediction (5-fold CV): multivariate regression MAE INR 15.0 lakh, median error 10%, R2 0.87; best model: Gradient boosting (MAE INR 11.7 lakh, median error 6%, R2 0.91).
- Drivers (regression, R2 0.89): 1% more area adds 1.23% to price; each extra bedroom -3.2%; resale vs new +2.4%; each extra listed amenity +0.4%; locality price level is the strongest driver (coef 1.07). Max VIF 3.7.
- Resale homes list at a median INR 5,609/sq ft vs INR 4,990 for new.
- Locality clusters (41 localities with 15+ listings, k = 5): mid-market, thin-supply, larger homes (10 localities, INR 5,101/sq ft); mid-market, high-supply (9 localities, INR 4,970/sq ft); affordable, high-supply (10 localities, INR 3,750/sq ft); premium, high-supply (6 localities, INR 7,997/sq ft); mid-market, thin-supply, smaller homes (6 localities, INR 4,874/sq ft).
- Gap, premium but thin supply (inventory to win): Madhapur (INR 9,005/sq ft, 31 listings), Sanath Nagar (INR 8,006/sq ft, 18 listings), Begumpet (INR 7,188/sq ft, 21 listings), Malkajgiri (INR 5,765/sq ft, 22 listings), Nallagandla Gachibowli (INR 5,749/sq ft, 32 listings).
- Volume hubs (lead volume, most competition): Kukatpally (164), Kondapur (148), Manikonda (113), Nizampet (106), Gachibowli (82).
