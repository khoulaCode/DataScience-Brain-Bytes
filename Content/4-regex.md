

##  **Data Cleaning with Regex in Python + Pandas**


##  **Why Use Regex for Data Cleaning?**

* Clean inconsistent formatting (e.g., phone numbers, dates).
* Remove unwanted characters (HTML tags, special symbols).
* Extract structured data (IDs, emails, product codes).
* Validate formats (email, postal codes).


##  **Common Regex + Pandas Operations**

| **Operation** | **Method**       | **Example**                                        | **Purpose**                     |
| ------------- | ---------------- | -------------------------------------------------- | ------------------------------- |
| Replace       | `str.replace()`  | `df['col'].str.replace(r'\d+', '', regex=True)`    | Remove digits                   |
| Extract       | `str.extract()`  | `df['col'].str.extract(r'(\d{4})')`                | Extract 4-digit year            |
| Match pattern | `str.match()`    | `df['col'].str.match(r'^\d{5}$')`                  | Check if value is a 5-digit zip |
| Filter rows   | `str.contains()` | `df[df['col'].str.contains(r'error', regex=True)]` | Rows containing 'error'         |
| Split strings | `str.split()`    | `df['col'].str.split(r'\s+')`                      | Split on whitespace             |


##  **Key Regex Metacharacters & Concepts**

| **Symbol / Concept** | **Meaning**                         | **Example** | **Matches**              |       |                |
| -------------------- | ----------------------------------- | ----------- | ------------------------ | ----- | -------------- |
| `.`                  | Any character except newline        | `a.b`       | `acb`, `a7b`             |       |                |
| `^`                  | Start of string                     | `^abc`      | `abc` but not `xabc`     |       |                |
| `$`                  | End of string                       | `abc$`      | `xxabc` but not `abcx`   |       |                |
| `*`                  | 0 or more of previous               | `ab*`       | `a`, `ab`, `abbb`        |       |                |
| `+`                  | 1 or more of previous               | `ab+`       | `ab`, `abbb` but not `a` |       |                |
| `?`                  | 0 or 1 of previous                  | `ab?`       | `a`, `ab`                |       |                |
| `{n}`                | Exactly n times                     | `a{3}`      | `aaa`                    |       |                |
| `{n,m}`              | Between n and m times               | `a{2,4}`    | `aa`, `aaa`, `aaaa`      |       |                |
| `[]`                 | Set of characters                   | `[aeiou]`   | any vowel                |       |                |
| `[^]`                | Negated set                         | `[^0-9]`    | anything except digits   |       |                |
| `()`                 | Grouping                            | `(abc)+`    | `abcabc`                 |       |                |
| \`                   | \`                                  | OR          | \`cat                    | dog\` | `cat` or `dog` |
| `\d`                 | Digit                               | `\d+`       | `123`                    |       |                |
| `\D`                 | Non-digit                           | `\D+`       | `abc`                    |       |                |
| `\w`                 | Word character (alphanumeric + `_`) | `\w+`       | `word_1`                 |       |                |
| `\W`                 | Non-word character                  | `\W+`       | `!@#`                    |       |                |
| `\s`                 | Whitespace                          | `\s+`       | space, tab               |       |                |
| `\S`                 | Non-whitespace                      | `\S+`       | `word`                   |       |                |


##  **Key Python `re` Module Methods**

| **Method**       | **Purpose**                           | **Example**                                         |
| ---------------- | ------------------------------------- | --------------------------------------------------- |
| `re.search()`    | Search for pattern anywhere in string | `re.search(r'\d+', 'abc123')` → match               |
| `re.match()`     | Match pattern at start of string      | `re.match(r'\d+', '123abc')` → match                |
| `re.fullmatch()` | Match whole string to pattern         | `re.fullmatch(r'\d+', '123')` → match               |
| `re.findall()`   | Find all occurrences                  | `re.findall(r'\d+', 'a12b34')` → `['12', '34']`     |
| `re.finditer()`  | Return iterator of matches            | `[m.group() for m in re.finditer(r'\w+', 'a b c')]` |
| `re.sub()`       | Replace matches                       | `re.sub(r'\d+', '', 'abc123')` → `'abc'`            |
| `re.split()`     | Split by pattern                      | `re.split(r'\s+', 'a b  c')` → `['a', 'b', 'c']`    |
| `re.compile()`   | Compile pattern for reuse             | `pattern = re.compile(r'\d+')`                      |


##  **Real Example with Pandas**

```python
data = {'raw': ['<p>Price: $100</p>', 'Cost is 200 USD', '€300']}
df = pd.DataFrame(data)

# Remove HTML tags
df['cleaned'] = df['raw'].str.replace(r'<.*?>', '', regex=True)

# Extract price numbers
df['price'] = df['cleaned'].str.extract(r'(\d+)').astype(float)

print(df)
```


##  **Tips for Using Regex in Pandas**

* Use `regex=True` when needed (especially in `str.replace()`).
* Chain `str.strip()`, `str.lower()` for normalization.
* Use `re.compile()` for complex or reused patterns.
* Use `na=False` in `str.contains()` to avoid errors on missing data:

  ```python
  df['col'].str.contains(r'pattern', regex=True, na=False)
  ```


