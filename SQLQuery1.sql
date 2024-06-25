select *
FROM credit_risk_dataset

DELETE
FROM credit_risk_dataset
WHERE loan_int_rate IS NULL

DELETE
FROM credit_risk_dataset
WHERE person_emp_length IS NULL

SELECT MAX(person_income) as max_income, 
		AVG(person_income) as avg_income
FROM credit_risk_dataset

