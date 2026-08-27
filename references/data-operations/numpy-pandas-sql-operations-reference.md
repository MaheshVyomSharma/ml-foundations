# NumPy — pandas — SQL Operations Reference

## 1. Introduction

This document provides a side-by-side reference for performing common data operations using **NumPy**, **pandas**, and **SQL**. The goal is to help learners quickly translate concepts between Python data analysis workflows and relational database queries. Each section demonstrates how the same task—such as filtering rows, aggregating data, or joining tables—is expressed in the three ecosystems.

### 1.1. How to read this comparison

The three tools work with data in related but different ways:

- **NumPy** works primarily with arrays, positions, masks, shapes, and vectorized numerical operations.
- **pandas** works with labelled tabular data using DataFrames and named columns.
- **SQL** describes operations on relational tables and returns unordered result sets unless an order is explicitly requested.

Some examples are therefore conceptual mappings rather than exact equivalents. In particular, NumPy does not provide native relational joins or grouped tabular operations, and generic SQL does not provide positional row slicing.

The document is structured as:

- a **sample dataset** used across examples
- a **core operations comparison section** for quick reference
- **appendices** containing best practices and conceptual mappings



## 2. Sample dataset

### 2.1. Employees
| emp_id | name  | dept_id | age | salary |
|---|---|---:|---:|---:|
| 101 | Asha  | 10 | 29 | 60000 |
| 102 | Ravi  | 20 | 35 | 75000 |
| 103 | Meera | 10 | 31 | 68000 |
| 104 | Kiran | 30 | 28 | 52000 |
| 105 | Zoya  | 20 | 41 | 82000 |

### 2.2. Departments
| dept_id | dept_name |
|---:|---|
| 10 | Engineering |
| 20 | HR |
| 30 | Finance |

### 2.3. Dependents
| dep_id | emp_id | dependent_name | relation |
|---:|---:|---|---|
| 1 | 101 | Anya  | Child |
| 2 | 102 | Neha  | Spouse |
| 3 | 102 | Rohan | Child |
| 4 | 105 | Sara  | Spouse |

---

## 3. Assumption for NumPy examples

`employees_np` is a **2D object array** with columns in this order:

- `0 = emp_id`
- `1 = name`
- `2 = dept_id`
- `3 = age`
- `4 = salary`

`employees_pd` is a pandas DataFrame with column names matching the Employees table.

The object array is used here to keep the comparison compact even though it contains both text and numeric values. In typical NumPy numerical work, homogeneous numeric arrays are preferable. The NumPy examples also use integer positions rather than column labels.

---

## 4. Load / create data

### 4.1. Mental note
- **NumPy:** create a 2D array, usually position-based.
- **pandas:** create a DataFrame with named columns.
- **SQL:** data is assumed to already exist in tables; loading usually means querying from an existing table.

### 4.2. NumPy
```python
employees_np = np.array([
    [101, "Asha", 10, 29, 60000],
    [102, "Ravi", 20, 35, 75000],
    [103, "Meera", 10, 31, 68000],
    [104, "Kiran", 30, 28, 52000],
    [105, "Zoya", 20, 41, 82000]
], dtype=object)
```

### 4.3. pandas
```python
employees_pd = pd.DataFrame({
    "emp_id": [101, 102, 103, 104, 105],
    "name": ["Asha", "Ravi", "Meera", "Kiran", "Zoya"],
    "dept_id": [10, 20, 10, 30, 20],
    "age": [29, 35, 31, 28, 41],
    "salary": [60000, 75000, 68000, 52000, 82000]
})
```

### 4.4. Reference representations for related tables

The later join and reshape examples use the following abbreviated representations. They correspond to the **Departments**, **Dependents**, and salary-by-year tables shown above. They are included to define the notation used in the comparisons, not as setup instructions for a runnable program.

For NumPy, the column positions are:

- `departments_np`: `0 = dept_id`, `1 = dept_name`
- `dependents_np`: `0 = dep_id`, `1 = emp_id`, `2 = dependent_name`, `3 = relation`
- `wide_np`: `0 = emp_id`, `1 = salary_2022`, `2 = salary_2023`

For pandas, the corresponding DataFrames are `departments_pd`, `dependents_pd`, and `salary_wide_pd`, with column names matching those tables.

The long-format pandas representation used for the reverse reshape is called `salary_long_pd`; it has the columns `emp_id`, `year`, and `salary`.

```python
departments_np = np.array([
    [10, "Engineering"],
    [20, "HR"],
    [30, "Finance"]
], dtype=object)

dependents_np = np.array([
    [1, 101, "Anya", "Child"],
    [2, 102, "Neha", "Spouse"],
    [3, 102, "Rohan", "Child"],
    [4, 105, "Sara", "Spouse"]
], dtype=object)

wide_np = np.array([
    [101, 58000, 60000],
    [102, 72000, 75000],
    [103, 65000, 68000]
], dtype=object)
```

```python
departments_pd = pd.DataFrame({
    "dept_id": [10, 20, 30],
    "dept_name": ["Engineering", "HR", "Finance"]
})

dependents_pd = pd.DataFrame({
    "dep_id": [1, 2, 3, 4],
    "emp_id": [101, 102, 102, 105],
    "dependent_name": ["Anya", "Neha", "Rohan", "Sara"],
    "relation": ["Child", "Spouse", "Child", "Spouse"]
})

salary_wide_pd = pd.DataFrame({
    "emp_id": [101, 102, 103],
    "salary_2022": [58000, 72000, 65000],
    "salary_2023": [60000, 75000, 68000]
})
```

### 4.5. SQL
```sql
SELECT *
FROM employees;
```

### 4.6. Shape note
- **NumPy:** 2D array with shape `(5, 5)`
- **pandas:** DataFrame with `5 rows × 5 columns`
- **SQL:** result set with `5 rows × 5 columns` for this sample data

---

## 5. Select rows and columns

### 5.1. Mental note
- **NumPy:** selection is by **integer position** using row and column indexes.
- **pandas:** selection can be **label-based** or **position-based**.
- **SQL:** selection is naturally column-based; row slicing by position is not part of core generic SQL.

### 5.2. Select columns: name and salary

#### 5.2.1. NumPy
```python
employees_np[:, [1, 4]]
```

#### 5.2.2. pandas
```python
employees_pd[["name", "salary"]]
```

#### 5.2.3. SQL
```sql
SELECT name, salary
FROM employees;
```

### 5.3. Shape note
- **NumPy:** shape `(5, 2)`
- **pandas:** `5 rows × 2 columns`
- **SQL:** result set with `5 rows × 2 columns`

### 5.4. Select rows: index 1 to 3

#### 5.4.1. NumPy
```python
employees_np[1:4]
```

#### 5.4.2. pandas
```python
employees_pd.iloc[1:4]
```

#### 5.4.3. SQL
```sql
-- No exact generic SQL equivalent by row index.
-- SQL tables are unordered unless an ORDER BY is applied.
```

### 5.5. Feasibility note
- **Why no exact SQL equivalent?** SQL works with unordered sets of rows unless an explicit ordering is defined. “Rows 1 to 3” is a positional idea that fits arrays/DataFrames more naturally than generic SQL.

### 5.6. Shape note
- **NumPy:** shape `(3, 5)`
- **pandas:** `3 rows × 5 columns`
- **SQL:** not applicable in generic form

### 5.7. Select rows and columns together
Example: rows `1:4`, columns `name` and `salary`

#### 5.7.1. NumPy
```python
employees_np[1:4, [1, 4]]
```

#### 5.7.2. pandas
```python
employees_pd.iloc[1:4][["name", "salary"]]
```

> **Note:** This chained selection is possible, but it is not preferred for more complex transformations. Each indexing step creates an intermediate result, which can make it unclear whether a later assignment changes the original DataFrame or only a temporary object. A single indexing operation communicates the intent more clearly and avoids potential chained-assignment warnings:

```python
employees_pd.iloc[1:4, [1, 4]]
```

When selecting by column names, label-based selection is often clearer:

```python
employees_pd.loc[1:3, ["name", "salary"]]
```

#### 5.7.3. SQL
```sql
-- No exact generic SQL equivalent for row positions 1:4.
-- Closest idea needs an explicit ordering plus row-number logic,
-- which goes beyond simple generic SQL reference style.
```

### 5.8. Feasibility note
- **Why only a partial SQL mapping?** Selecting columns is easy in SQL, but selecting a positional row slice is not a clean generic equivalent unless we introduce extra constructs such as ordering and row numbering.

### 5.9. Shape note
- **NumPy:** shape `(3, 2)`
- **pandas:** `3 rows × 2 columns`
- **SQL:** not applicable in clean generic form

---

## 6. Filter rows by condition

### 6.1. Mental note
- **NumPy:** filtering is usually done with a boolean mask.
- **pandas:** filtering also uses a boolean mask, but column access is more readable.
- **SQL:** filtering is done with `WHERE`.

### 6.2. Filter employees with salary > 65000

#### 6.2.1. NumPy
```python
employees_np[employees_np[:, 4] > 65000]
```

#### 6.2.2. pandas
```python
employees_pd[employees_pd["salary"] > 65000]
```

#### 6.2.3. SQL
```sql
SELECT *
FROM employees
WHERE salary > 65000;
```

### 6.3. Shape note
- **NumPy:** shape `(3, 5)` for this sample data
- **pandas:** `3 rows × 5 columns` for this sample data
- **SQL:** result set with `3 rows × 5 columns` for this sample data

### 6.4. Filter employees in dept_id == 10

#### 6.4.1. NumPy
```python
employees_np[employees_np[:, 2] == 10]
```

#### 6.4.2. pandas
```python
employees_pd[employees_pd["dept_id"] == 10]
```

#### 6.4.3. SQL
```sql
SELECT *
FROM employees
WHERE dept_id = 10;
```

### 6.5. Shape note
- **NumPy:** shape `(2, 5)` for this sample data
- **pandas:** `2 rows × 5 columns` for this sample data
- **SQL:** result set with `2 rows × 5 columns` for this sample data

### 6.6. Filter with multiple conditions
Example: `dept_id == 20` and `age > 35`

#### 6.6.1. NumPy
```python
employees_np[(employees_np[:, 2] == 20) & (employees_np[:, 3] > 35)]
```

#### 6.6.2. pandas
```python
employees_pd[(employees_pd["dept_id"] == 20) & (employees_pd["age"] > 35)]
```

#### 6.6.3. SQL
```sql
SELECT *
FROM employees
WHERE dept_id = 20 AND age > 35;
```

### 6.7. Shape note
- **NumPy:** shape `(1, 5)` for this sample data
- **pandas:** `1 row × 5 columns` for this sample data
- **SQL:** result set with `1 row × 5 columns` for this sample data


---

## 7. Sort data

### 7.1. Mental note
- **NumPy:** sorting typically uses `np.argsort()` to reorder rows.
- **pandas:** sorting is straightforward with `sort_values()`.
- **SQL:** sorting is done with `ORDER BY`.

### 7.2. Sort employees by salary (ascending)

#### 7.2.1. NumPy
```python
employees_np[np.argsort(employees_np[:, 4])]
```

#### 7.2.2. pandas
```python
employees_pd.sort_values("salary")
```

#### 7.2.3. SQL
```sql
SELECT *
FROM employees
ORDER BY salary ASC;
```

### 7.3. Shape note
- **NumPy:** shape `(5, 5)` (rows reordered)
- **pandas:** `5 rows × 5 columns`
- **SQL:** result set with `5 rows × 5 columns`

### 7.4. Sort employees by age (descending)

#### 7.4.1. NumPy
```python
employees_np[np.argsort(-employees_np[:, 3].astype(int))]
```

#### 7.4.2. pandas
```python
employees_pd.sort_values("age", ascending=False)
```

#### 7.4.3. SQL
```sql
SELECT *
FROM employees
ORDER BY age DESC;
```

---

## 8. Aggregation / summary statistics

### 8.1. Mental note
- **NumPy:** aggregation functions operate directly on arrays.
- **pandas:** aggregation operates on Series or DataFrame columns.
- **SQL:** aggregation uses functions like `AVG`, `SUM`, `COUNT`.

### 8.2. Average salary

#### 8.2.1. NumPy
```python
np.mean(employees_np[:, 4].astype(int))
```

#### 8.2.2. pandas
```python
employees_pd["salary"].mean()
```

#### 8.2.3. SQL
```sql
SELECT AVG(salary)
FROM employees;
```

### 8.3. Total salary expense

#### 8.3.1. NumPy
```python
np.sum(employees_np[:, 4].astype(int))
```

#### 8.3.2. pandas
```python
employees_pd["salary"].sum()
```

#### 8.3.3. SQL
```sql
SELECT SUM(salary)
FROM employees;
```

### 8.4. Employee count

#### 8.4.1. NumPy
```python
employees_np.shape[0]
```

#### 8.4.2. pandas
```python
employees_pd.shape[0]
```

#### 8.4.3. SQL
```sql
SELECT COUNT(*)
FROM employees;
```

---

## 9. Group by

### 9.1. Mental note
- **NumPy:** grouping is not native; typically requires masking or helper logic.
- **pandas:** `groupby()` provides powerful grouped operations.
- **SQL:** grouping uses `GROUP BY`.

### 9.2. Average salary per department

#### 9.2.1. NumPy
```python
for dept_id in np.unique(employees_np[:, 2]):
    salaries = employees_np[employees_np[:, 2] == dept_id][:, 4].astype(int)
    print(dept_id, np.mean(salaries))
```

*(NumPy requires explicit masking and iteration over the groups.)*

#### 9.2.2. pandas
```python
employees_pd.groupby("dept_id")["salary"].mean()
```

#### 9.2.3. SQL
```sql
SELECT dept_id, AVG(salary)
FROM employees
GROUP BY dept_id;
```

### 9.3. Employee count per department

#### 9.3.1. NumPy
```python
dept_ids, counts = np.unique(employees_np[:, 2], return_counts=True)
list(zip(dept_ids, counts))
```

*(NumPy can calculate the counts, but it does not provide a table-oriented `groupby()` abstraction.)*

#### 9.3.2. pandas
```python
employees_pd.groupby("dept_id").size()
```

#### 9.3.3. SQL
```sql
SELECT dept_id, COUNT(*)
FROM employees
GROUP BY dept_id;
```

For the sample data, both approaches produce the following grouped results:

| dept_id | average salary | employee count |
|---:|---:|---:|
| 10 | 64000 | 2 |
| 20 | 78500 | 2 |
| 30 | 52000 | 1 |


---

## 10. Merge / Join tables

### 10.1. Mental note
- **NumPy:** no built-in relational join capability; joins require manual matching logic.
- **pandas:** `merge()` provides SQL-like joins directly on DataFrames.
- **SQL:** joins are a core relational operation using `JOIN` clauses.

### 10.2. Join employees with departments (get department name for each employee)

#### 10.2.1. NumPy
```python
# Conceptual example (manual lookup approach)
result = []
for emp in employees_np:
    dept = departments_np[departments_np[:,0] == emp[2]][0]
    result.append(list(emp) + [dept[1]])

np.array(result, dtype=object)
```

*(Requires manual matching on dept_id; no built‑in join operation.)*

#### 10.2.2. pandas
```python
employees_pd.merge(departments_pd, on="dept_id")
```

#### 10.2.3. SQL
```sql
SELECT e.*, d.dept_name
FROM employees e
JOIN departments d
ON e.dept_id = d.dept_id;
```

### 10.3. Shape note
- **NumPy:** shape `(5, 6)` after appending department name
- **pandas:** `5 rows × 6 columns`
- **SQL:** result set with `5 rows × 6 columns`

---

### 10.4. Join employees with dependents

*(Shows one‑to‑many relationship: one employee may have multiple dependents.)*

#### 10.4.1. NumPy
```python
# Conceptual manual join
result = []
for emp in employees_np:
    matches = dependents_np[dependents_np[:,1] == emp[0]]
    for dep in matches:
        result.append(list(emp) + list(dep))

np.array(result, dtype=object)
```

*(Requires nested loops and manual matching logic.)*

#### 10.4.2. pandas
```python
employees_pd.merge(dependents_pd, on="emp_id")
```

#### 10.4.3. SQL
```sql
SELECT e.*, d.dependent_name, d.relation
FROM employees e
JOIN dependents d
ON e.emp_id = d.emp_id;
```

### 10.5. Shape note
- **NumPy:** number of rows expands depending on dependents
- **pandas:** rows expand for one‑to‑many matches
- **SQL:** result set expands similarly

---

### 10.6. Left join example

Get **all employees**, even if they have **no dependents**.

#### 10.6.1. NumPy
```python
# Requires custom logic to insert rows when no match exists
# No native left join functionality
```

#### 10.6.2. pandas
```python
employees_pd.merge(dependents_pd, on="emp_id", how="left")
```

#### 10.6.3. SQL
```sql
SELECT e.*, d.dependent_name, d.relation
FROM employees e
LEFT JOIN dependents d
ON e.emp_id = d.emp_id;
```

### 10.7. Mental takeaway
- **NumPy:** joins are cumbersome and not idiomatic.
- **pandas:** designed to handle relational-style data analysis.
- **SQL:** native environment for relational joins.

---

## 11. Handling missing values

### 11.1. Mental note
- **NumPy:** missing numeric values are commonly represented by `np.nan`, a floating-point value that represents “not a number.”
- **pandas:** uses missing-value markers such as `NaN` and `pd.NA`, with functions such as `isna()`, `dropna()`, and `fillna()` to handle them consistently across column types.
- **SQL:** missing values are represented by `NULL`, which means that a value is absent or unknown. `NULL` is not the same as zero, an empty string, or `FALSE`, and comparisons with it use `IS NULL` or `IS NOT NULL` rather than `=`.

These representations are conceptually related but are not interchangeable. Their propagation rules, comparisons, and supported types depend on the tool, so a missing-value operation should be interpreted in the context of the relevant system.

### 11.2. Detect missing values in salary

#### 11.2.1. NumPy
```python
np.isnan(employees_np[:, 4].astype(float))
```

#### 11.2.2. pandas
```python
employees_pd["salary"].isna()
```

#### 11.2.3. SQL
```sql
SELECT *
FROM employees
WHERE salary IS NULL;
```

### 11.3. Replace missing salary with 0

#### 11.3.1. NumPy
```python
col = employees_np[:, 4].astype(float)
col[np.isnan(col)] = 0
```

#### 11.3.2. pandas
```python
employees_pd["salary"] = employees_pd["salary"].fillna(0)
```

`fillna(0)` returns a filled Series; assigning it back updates the DataFrame.

#### 11.3.3. SQL
```sql
SELECT emp_id, name, COALESCE(salary, 0) AS salary
FROM employees;
```

### 11.4. Shape note
Handling missing values does not change row/column count unless rows are dropped.

---

## 12. Creating derived / new columns

### 12.1. Mental note
- **NumPy:** new columns require constructing a new array or stacking columns.
- **pandas:** very easy via vectorized column assignment.
- **SQL:** new columns are typically created in the `SELECT` clause using expressions.

### 12.2. Create a new column: salary in thousands

#### 12.2.1. NumPy
```python
salary_k = employees_np[:, 4].astype(int) / 1000
np.column_stack((employees_np, salary_k))
```

#### 12.2.2. pandas
```python
employees_pd["salary_k"] = employees_pd["salary"] / 1000
```

#### 12.2.3. SQL
```sql
SELECT *, salary/1000 AS salary_k
FROM employees;
```

### 12.3. Shape note
- **NumPy:** shape becomes `(5, 6)` after stacking new column
- **pandas:** DataFrame becomes `5 rows × 6 columns`
- **SQL:** result set includes the computed column but does not modify the base table

---

## 13. Wide ↔ Long format conversion

### 13.1. Mental note
- **NumPy:** no dedicated reshape utilities for tabular pivoting; requires manual restructuring.
- **pandas:** built-in tools like `melt()` and `pivot()` handle this cleanly.
- **SQL:** can approximate using `UNPIVOT`, `PIVOT`, or `CASE` depending on the database.

### 13.2. Example wide dataset (salary by year)

| emp_id | salary_2022 | salary_2023 |
|---|---|---|
| 101 | 58000 | 60000 |
| 102 | 72000 | 75000 |
| 103 | 65000 | 68000 |

---

### 13.3. Wide → Long

Transform yearly salary columns into rows.

#### 13.3.1. NumPy
```python
# Conceptual approach: manually rebuild rows
rows = []
for r in wide_np:
    rows.append([r[0], "2022", r[1]])
    rows.append([r[0], "2023", r[2]])

np.array(rows, dtype=object)
```

*(Requires manual restructuring logic.)*

#### 13.3.2. pandas
```python
pd.melt(
    salary_wide_pd,
    id_vars="emp_id",
    var_name="year",
    value_name="salary"
)
```

#### 13.3.3. SQL
```sql
SELECT emp_id, '2022' AS year, salary_2022 AS salary FROM salary_table
UNION ALL
SELECT emp_id, '2023' AS year, salary_2023 AS salary FROM salary_table;
```

### 13.4. Shape note
- **NumPy:** rows increase because columns are converted into rows.
- **pandas:** `melt()` expands rows accordingly.
- **SQL:** `UNION` or `UNPIVOT` produces the long format.

---

### 13.5. Long → Wide

Convert row-based years back into columns.

#### 13.5.1. NumPy
```python
# Requires custom grouping and reconstruction logic
```

#### 13.5.2. pandas
```python
salary_long_pd.pivot(
    index="emp_id",
    columns="year",
    values="salary"
)
```

#### 13.5.3. SQL
```sql
SELECT emp_id,
MAX(CASE WHEN year='2022' THEN salary END) AS salary_2022,
MAX(CASE WHEN year='2023' THEN salary END) AS salary_2023
FROM salary_long
GROUP BY emp_id;
```

### 13.6. Mental takeaway
- **NumPy:** awkward for reshaping relational-style datasets.
- **pandas:** designed for this with `melt`, `pivot`, and `pivot_table`.
- **SQL:** possible but verbose without dedicated `PIVOT/UNPIVOT` support.

---

## 14. Handling duplicates

### 14.1. Mental note
- **NumPy:** requires manual comparison or use of helper functions like `np.unique()`.
- **pandas:** provides `drop_duplicates()` and `duplicated()`.
- **SQL:** uses `DISTINCT` or grouping to remove duplicate rows.

### 14.2. Remove duplicate rows

#### 14.2.1. NumPy
```python
np.unique(employees_np, axis=0)
```

#### 14.2.2. pandas
```python
employees_pd.drop_duplicates()
```

#### 14.2.3. SQL
```sql
SELECT DISTINCT *
FROM employees;
```

### 14.3. Shape note
Removing duplicates may reduce the number of rows.

---

## 15. Type conversion

### 15.1. Mental note
- **NumPy:** uses `astype()` for array type casting.
- **pandas:** also uses `astype()` but can operate on individual columns.
- **SQL:** uses `CAST()` or `CONVERT()` depending on the database.

### 15.2. Convert salary to integer

#### 15.2.1. NumPy
```python
employees_np[:,4].astype(int)
```

#### 15.2.2. pandas
```python
employees_pd["salary"].astype(int)
```

#### 15.2.3. SQL
```sql
SELECT CAST(salary AS INT)
FROM employees;
```

### 15.3. Shape note
Type conversion does not change the dataset shape.

---

## 16. String operations

### 16.1. Mental note
- **NumPy:** limited string processing utilities.
- **pandas:** powerful vectorized string methods via `.str`.
- **SQL:** uses built-in string functions.

### 16.2. Convert employee names to uppercase

#### 16.2.1. NumPy
```python
np.char.upper(employees_np[:,1])
```

#### 16.2.2. pandas
```python
employees_pd["name"].str.upper()
```

#### 16.2.3. SQL
```sql
SELECT UPPER(name)
FROM employees;
```

### 16.3. Shape note
String transformations modify values but do not change dataset shape.

---

## 17. Best practices

### 17.1. pandas row and column selection

Simplified example used earlier:

```python
employees_pd.iloc[1:4][["name","salary"]]
```

Preferred approach:

```python
employees_pd.loc[1:3, ["name","salary"]]
```

Reason:
- avoids chained indexing
- clearer semantics

The chained form remains valid for selection, but the single-step form is preferred because it makes the target rows and columns explicit in one operation and avoids ambiguity when the result is later modified.

---

### 17.2. NumPy sorting with argsort

```python
employees_np[np.argsort(employees_np[:,4])]
```

Explanation:
- `argsort()` returns the index order of sorted values
- allows row-wise reordering

---

### 17.3. SQL row ordering

SQL tables have **no guaranteed row order**.

Correct pattern:

```sql
SELECT *
FROM employees
ORDER BY salary;
```

---

## 18. Concept mapping

| Concept | NumPy | pandas | SQL |
|---|---|---|---|
| Table | 2D array | DataFrame | Table |
| Column | array slice | Series | Column |
| Row | array row | row | row |
| Index / identity | position | index | primary key |

---

## 19. SQL caveats

SQL syntax and behavior can vary by database vendor. The examples in this document use broadly recognizable SQL, but they should be checked against the target database before being used in production.

| Feature | ANSI SQL | Vendor specific |
|---|---|---|
| PIVOT | not standard | SQL Server / Oracle |
| LIMIT | MySQL/Postgres | not ANSI |
| TOP | SQL Server | not portable |
