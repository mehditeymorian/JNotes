# Jaccord Similarity & Norm Similarity
this notebook contains code to calculate jaccord and L1 L2 similarity for stackoverflow users based on their votes on different questions.

# Data
Dataset is a xml file consists of vote record. [Link to Dataset](https://archive.org/download/stackexchange/stackoverflow.com-Votes.7z)
```xml
<row Id="1" PostId="1" VoteTypeId="2" CreationDate="2008-07-31T00:00:00.000" />
...
<row Id="2342341" PostId="234534652" VoteTypeId="8" CreationDate="2008-07-31T00:00:00.000" />
```

# Clean Data
dataset is a big chunk of data and cannot be loaded in memory at once. we can achieve this by using Linux pipeline and commands include cat, grep, perl at file level.
In this script, we read content of file and pass it to grep command to select those record which has `UserId` attribute. Finally by using perl, we can use REGEX to extract groups of data from each line. at the end, we write the result to a text file.
```bash
cat votes.xml | grep "UserId" | perl -ne 'print "$1 $2 $3 $4\n" if /Id="(\d*)"\sPostId="(\d*)"\sVoteTypeId="(\d*)"\sUserId="(\d*)"/ ' >> result.txt
```
Dataset size shrunk to 390MB, approximately 1/53 of the initial size. Now the dataset look like this:
```
Id        PostId   VoteTypeId   UserId
175053113 42879012 5 5304980
175053135 44499101 5 656208
175053143 82831 5 5418727
175053158 30237702 5 2796523
175053184 4277665 5 5446749
...
```


```python
import pandas as pd
import numpy as np
```


```python
# reading the data file and labeling each column
data = pd.read_csv(filepath_or_buffer='data.txt', sep=' ', names=['voteId', 'postId', 'voteType', 'userId'])
```


```python
# group data by userId and sort them by their len descending.
groups_by_userId = data.groupby('userId')
users = groups_by_userId.userId.agg([len]).sort_values(by='len', ascending=False)
# these are the top 5 users who vote more than others.
users.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>len</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6309</th>
      <td>37000</td>
    </tr>
    <tr>
      <th>432903</th>
      <td>19704</td>
    </tr>
    <tr>
      <th>3475600</th>
      <td>15479</td>
    </tr>
    <tr>
      <th>850848</th>
      <td>14830</td>
    </tr>
    <tr>
      <th>1415724</th>
      <td>14017</td>
    </tr>
  </tbody>
</table>
</div>




```python
# similarities are only calculated for top 5 users.
ids = np.array([6309, 432903, 3475600, 850848, 1415724])
# filter data base on top 5 users.
filtered_data = data[data.userId.isin(ids)]

# used to replace initial vote type with 1 0 and -1
# 5 -> 1
# 8 -> -1
# nan -> 0
def agg_func(x):
    t = sum(x)
    return +1 if t == 5 else -1 if t == 8 else 0


# generate junction table of user and post with each cell holding user vote on that post.
#           Post_j Post_j+1 ...
# User_i    1      0
# User_i+1  0      -1
# User_i+2  -1      1
table = pd.crosstab(filtered_data.postId, filtered_data.userId, values=filtered_data.voteType, aggfunc=agg_func)
```

# Jaccord Similarity
The Jaccard index, also known as the Jaccard similarity coefficient, is a statistic used for gauging the similarity and diversity of sample sets.

![Jaccord Similarity Formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/eaef5aa86949f49e7dc6b9c8c3dd8b233332c9e7)


```python
def jaccord_similarity(x, y):
    # calculate the n and U
    n = np.nansum(np.transpose(x) * np.array(y))
    u = np.nansum(x) + np.nansum(y) - n
    # base on the formula
    similarity = n / u
    distance =  1 - similarity
    return [similarity, distance]
```

# Cosine L norm Similarity
Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them, which is also the same as the inner product of the same vectors normalized to both have length 1. From the latter definition, it follows that the cosine similarity depends only on the angle between the two non-zero vectors, but not on their magnitudes.

![Cosine Similarity](https://wikimedia.org/api/rest_v1/media/math/render/svg/0a4c9a778656537624a3303e646559a429868863)

# L Normalization
Another effective proxy for cosine distance can be obtained by {\displaystyle L_{2}}L_{2} normalisation of the vectors, followed by the application of normal Euclidean distance. Using this technique each term in each vector is first divided by the magnitude of the vector, yielding a vector of unit length. Then, it is clear, the Euclidean distance over the end-points of any two vectors is a proper metric which gives the same ordering as the cosine distance for any comparison of vectors, and furthermore avoids the potentially expensive trigonometric operations required to yield a proper metric. Once the normalisation has occurred, the vector space can be used with the full range of techniques available to any Euclidean space, notably standard dimensionality reduction techniques. This normalised form distance is often used within many deep learning algorithms.

![L Norm](https://wikimedia.org/api/rest_v1/media/math/render/svg/53a5615d02f7a03013e22bd4adf055cdbe4a303c)



```python
# calculate the L norm for a vector
def l_norm(l, x):
    sum_numbers = 0
    for i in x:
        # for element not nan elements
        if not pd.isna(i):
            sum_numbers += i ** l
    # based on the formula
    return pow(sum_numbers, 1 / l)

# calculate the cosine l norm similarity for 2 vectors of x and y
def norm_similarity(l, x, y):
    similarity = np.nansum(x * y) / (l_norm(l, x) * l_norm(l, y))
    distance = 1.0 - similarity
    return [similarity, distance]
```


```python
# similarity of pairs
jaccord_result = []
l1norm_result = []
l2norm_result = []

# this loops generate pair for top 5 users
for i in ids:
    for j in ids:
        if i != j:
            first = table.get(i)
            second = table.get(j)
            # add jaccord similarity
            jaccord = jaccord_similarity(first, second)
            jaccord_result.append([f"{i},{j}", jaccord[0], jaccord[1]])
            # add Cosine L1Norm similarity
            l1norm = norm_similarity(1,first, second)
            l1norm_result.append([f"{i},{j}", l1norm[0], l1norm[1]])
            # add Cosine L2Norm similarity
            l2norm = norm_similarity(2,first, second)
            l2norm_result.append([f"{i},{j}", l2norm[0], l2norm[1]])

```

# Jaccord Similarity


```python
pd.DataFrame(jaccord_result, columns=['pair', 'similarity', 'distance'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pair</th>
      <th>similarity</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6309,432903</td>
      <td>0.005640</td>
      <td>0.994360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6309,3475600</td>
      <td>0.000534</td>
      <td>0.999466</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6309,850848</td>
      <td>0.000696</td>
      <td>0.999304</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6309,1415724</td>
      <td>0.000039</td>
      <td>0.999961</td>
    </tr>
    <tr>
      <th>4</th>
      <td>432903,6309</td>
      <td>0.005640</td>
      <td>0.994360</td>
    </tr>
    <tr>
      <th>5</th>
      <td>432903,3475600</td>
      <td>0.003594</td>
      <td>0.996406</td>
    </tr>
    <tr>
      <th>6</th>
      <td>432903,850848</td>
      <td>0.000551</td>
      <td>0.999449</td>
    </tr>
    <tr>
      <th>7</th>
      <td>432903,1415724</td>
      <td>0.001425</td>
      <td>0.998575</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3475600,6309</td>
      <td>0.000534</td>
      <td>0.999466</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3475600,432903</td>
      <td>0.003594</td>
      <td>0.996406</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3475600,850848</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3475600,1415724</td>
      <td>0.000068</td>
      <td>0.999932</td>
    </tr>
    <tr>
      <th>12</th>
      <td>850848,6309</td>
      <td>0.000696</td>
      <td>0.999304</td>
    </tr>
    <tr>
      <th>13</th>
      <td>850848,432903</td>
      <td>0.000551</td>
      <td>0.999449</td>
    </tr>
    <tr>
      <th>14</th>
      <td>850848,3475600</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>850848,1415724</td>
      <td>0.000069</td>
      <td>0.999931</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1415724,6309</td>
      <td>0.000039</td>
      <td>0.999961</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1415724,432903</td>
      <td>0.001425</td>
      <td>0.998575</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1415724,3475600</td>
      <td>0.000068</td>
      <td>0.999932</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1415724,850848</td>
      <td>0.000069</td>
      <td>0.999931</td>
    </tr>
  </tbody>
</table>
</div>



# Cosine L1 Norm Similarity


```python
pd.DataFrame(l1norm_result, columns=['pair', 'similarity', 'distance'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pair</th>
      <th>similarity</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6309,432903</td>
      <td>4.362324e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6309,3475600</td>
      <td>4.889454e-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6309,850848</td>
      <td>6.581078e-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6309,1415724</td>
      <td>3.856738e-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>432903,6309</td>
      <td>4.362324e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>432903,3475600</td>
      <td>4.131172e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>432903,850848</td>
      <td>6.521515e-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>432903,1415724</td>
      <td>1.737928e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3475600,6309</td>
      <td>4.889454e-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3475600,432903</td>
      <td>4.131172e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3475600,850848</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3475600,1415724</td>
      <td>9.217901e-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>850848,6309</td>
      <td>6.581078e-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>850848,432903</td>
      <td>6.521515e-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>850848,3475600</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>850848,1415724</td>
      <td>9.649931e-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1415724,6309</td>
      <td>3.856738e-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1415724,432903</td>
      <td>1.737928e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1415724,3475600</td>
      <td>9.217901e-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1415724,850848</td>
      <td>9.649931e-09</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# Cosine L2 Norm Similarity


```python
pd.DataFrame(l2norm_result, columns=['pair', 'similarity', 'distance'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pair</th>
      <th>similarity</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6309,432903</td>
      <td>0.011778</td>
      <td>0.988222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6309,3475600</td>
      <td>0.001170</td>
      <td>0.998830</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6309,850848</td>
      <td>0.001538</td>
      <td>0.998462</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6309,1415724</td>
      <td>0.000088</td>
      <td>0.999912</td>
    </tr>
    <tr>
      <th>4</th>
      <td>432903,6309</td>
      <td>0.011778</td>
      <td>0.988222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>432903,3475600</td>
      <td>0.007215</td>
      <td>0.992785</td>
    </tr>
    <tr>
      <th>6</th>
      <td>432903,850848</td>
      <td>0.001112</td>
      <td>0.998888</td>
    </tr>
    <tr>
      <th>7</th>
      <td>432903,1415724</td>
      <td>0.002888</td>
      <td>0.997112</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3475600,6309</td>
      <td>0.001170</td>
      <td>0.998830</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3475600,432903</td>
      <td>0.007215</td>
      <td>0.992785</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3475600,850848</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3475600,1415724</td>
      <td>0.000136</td>
      <td>0.999864</td>
    </tr>
    <tr>
      <th>12</th>
      <td>850848,6309</td>
      <td>0.001538</td>
      <td>0.998462</td>
    </tr>
    <tr>
      <th>13</th>
      <td>850848,432903</td>
      <td>0.001112</td>
      <td>0.998888</td>
    </tr>
    <tr>
      <th>14</th>
      <td>850848,3475600</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>850848,1415724</td>
      <td>0.000139</td>
      <td>0.999861</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1415724,6309</td>
      <td>0.000088</td>
      <td>0.999912</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1415724,432903</td>
      <td>0.002888</td>
      <td>0.997112</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1415724,3475600</td>
      <td>0.000136</td>
      <td>0.999864</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1415724,850848</td>
      <td>0.000139</td>
      <td>0.999861</td>
    </tr>
  </tbody>
</table>
</div>


