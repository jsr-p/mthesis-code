import io

import pandas as pd

data = io.StringIO(
    """a,b,c,d,e,f,g,h,i,2.0
    1,2.5,True,a,,,,,
    3,4.5,False,b,6,7.5,True,a,3.0
"""
)


df_pyarrow = pd.read_csv(data, dtype_backend="pyarrow")  # type: ignore

print(df_pyarrow.dtypes)
print(df_pyarrow)
