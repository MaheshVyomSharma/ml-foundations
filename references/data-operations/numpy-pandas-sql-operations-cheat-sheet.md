# NumPy — pandas — SQL Operations Cheat Sheet

## 1. Introduction

This cheat sheet provides a compact side-by-side reference for common data operations using **NumPy**, **pandas**, and **SQL**. It is intended for quick lookup after the concepts have been introduced in the [NumPy, pandas, and SQL Operations Reference](numpy_pandas_sql_operations_reference.md).

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

---

## 4. Core operations — NumPy / pandas / SQL comparison

This section compresses the core operations into side-by-side comparisons for quick lookup.

### 4.1. Load / create data

| NumPy | pandas | SQL |
|---|---|---|
| `np.array([...], dtype=object)` | `pd.DataFrame({...})` | `SELECT * FROM employees;` |

---

### 4.2. Select rows and columns

| Task | NumPy | pandas | SQL |
|---|---|---|---|
| Select columns | `employees_np[:, [1,4]]` | `employees_pd[["name","salary"]]` | `SELECT name, salary FROM employees;` |
| Select rows | `employees_np[1:4]` | `employees_pd.iloc[1:4]` | Not positional in generic SQL |
| Rows + columns | `employees_np[1:4,[1,4]]` | `employees_pd.iloc[1:4][["name","salary"]]` | Requires ordering + row numbering |

---

### 4.3. Filter rows

| NumPy | pandas | SQL |
|---|---|---|
| `employees_np[employees_np[:,4] > 65000]` | `employees_pd[employees_pd["salary"] > 65000]` | `SELECT * FROM employees WHERE salary > 65000;` |

---

### 4.4. Sort data

| NumPy | pandas | SQL |
|---|---|---|
| `employees_np[np.argsort(employees_np[:,4])]` | `employees_pd.sort_values("salary")` | `SELECT * FROM employees ORDER BY salary;` |

---

### 4.5. Aggregation

| Task | NumPy | pandas | SQL |
|---|---|---|---|
| Average salary | `np.mean(employees_np[:,4].astype(int))` | `employees_pd["salary"].mean()` | `SELECT AVG(salary) FROM employees;` |
| Total salary | `np.sum(employees_np[:,4].astype(int))` | `employees_pd["salary"].sum()` | `SELECT SUM(salary) FROM employees;` |
| Row count | `employees_np.shape[0]` | `employees_pd.shape[0]` | `SELECT COUNT(*) FROM employees;` |

---

### 4.6. Group by

| NumPy | pandas | SQL |
|---|---|---|
| Manual masking per group | `employees_pd.groupby("dept_id")["salary"].mean()` | `SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id;` |

---

### 4.7. Merge / join

| NumPy | pandas | SQL |
|---|---|---|
| Manual matching logic | `employees_pd.merge(departments_pd, on="dept_id")` | `SELECT e.*, d.dept_name FROM employees e JOIN departments d ON e.dept_id=d.dept_id;` |

---

### 4.8. Handling missing values

| NumPy | pandas | SQL |
|---|---|---|
| `np.isnan(col)` | `employees_pd["salary"].isna()` | `WHERE salary IS NULL` |
| Replace missing | manual assignment | `fillna(0)` | `COALESCE(salary, 0)` |

---

### 4.9. Derived columns

| NumPy | pandas | SQL |
|---|---|---|
| `np.column_stack((employees_np, salary_k))` | `employees_pd["salary_k"] = employees_pd["salary"]/1000` | `SELECT *, salary/1000 AS salary_k FROM employees;` |

---

### 4.10. Wide ↔ Long reshape

| Transformation | NumPy | pandas | SQL |
|---|---|---|---|
| Wide → Long | manual reconstruction | `pd.melt()` | `UNION ALL` or `UNPIVOT` |
| Long → Wide | manual reconstruction | `pivot()` | `CASE + GROUP BY` |

---

### 4.11. Handling duplicates

| NumPy | pandas | SQL |
|---|---|---|
| `np.unique(arr, axis=0)` | `drop_duplicates()` | `SELECT DISTINCT *` |

---

### 4.12. Type conversion

| NumPy | pandas | SQL |
|---|---|---|
| `astype(int)` | `astype(int)` | `CAST(column AS INT)` |

---

### 4.13. String operations

| NumPy | pandas | SQL |
|---|---|---|
| `np.char.upper()` | `.str.upper()` | `UPPER(column)` |

---

## 5. Summary

Although **NumPy**, **pandas**, and **SQL** operate in different environments, they share many conceptual similarities when working with tabular data. NumPy focuses on efficient numerical arrays, pandas provides powerful data manipulation tools for analysis, and SQL offers declarative querying for relational databases. Understanding how common operations map across these systems allows practitioners to move more easily between data science workflows and database querying.

---

## 6. Best practices

### 6.1. pandas row and column selection

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

The chained form is possible for selection, but the single-step form is preferred because it avoids intermediate results and makes later assignments less ambiguous.

---

### 6.2. NumPy sorting with argsort

```python
employees_np[np.argsort(employees_np[:,4])]
```

Explanation:
- `argsort()` returns the index order of sorted values
- allows row-wise reordering

---

### 6.3. SQL row ordering

SQL tables have **no guaranteed row order**.

Correct pattern:

```sql
SELECT *
FROM employees
ORDER BY salary;
```

---

## 7. Concept mapping

| Concept | NumPy | pandas | SQL |
|---|---|---|---|
| Table | 2D array | DataFrame | Table |
| Column | array slice | Series | Column |
| Row | array row | row | row |
| Index / identity | position | index | primary key |

---

## 8. SQL caveats

SQL syntax and behavior can vary by database vendor. Check database-specific features before using them in production.

| Feature | ANSI SQL | Vendor specific |
|---|---|---|
| PIVOT | not standard | SQL Server / Oracle |
| LIMIT | MySQL/Postgres | not ANSI |
| TOP | SQL Server | not portable |
