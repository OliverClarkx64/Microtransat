import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Lift_0 = pd.read_excel("2D_NACA0012_polars.xlsx", sheet_name="Cl", index_col=0)
Drag_0 = pd.read_excel("2D_NACA0012_polars.xlsx", sheet_name="Cd0", index_col=0)



ang_resample = np.concatenate((np.arange(0, 30, 1), np.arange(30, 150, 5), np.arange(150, 181, 1)))

print(ang_resample)

Polars = Lift_0.reindex(Lift_0.index.union(ang_resample)).interpolate('index').loc[ang_resample]
Drag_resampled = Drag_0.reindex(Drag_0.index.union(ang_resample)).interpolate('index').loc[ang_resample]
Polars = Polars.join(Drag_resampled)

angs_2 = np.flip( np.concatenate((np.arange(180, 210, 1), np.arange(210, 330, 5), np.arange(330, 361, 1)))  )

Polars_2 = pd.DataFrame(index = Polars.index)
Polars_2["ang_2"] = angs_2
Polars_2["Cl"] = -Polars["Cl"]
Polars_2["Cd0"] = Polars["Cd0"]
Polars_2 = Polars_2.sort_values(by = "ang_2").set_index("ang_2").iloc[1:]
#print(Polars_2)
Polars_3 = pd.concat([Polars, Polars_2])



filename = "clean_polars.xlsx"
Polars_3.to_excel(filename)

plt.plot(Lift_0, '-', label='Lift')
plt.plot(Drag_0, '-', label='Drag')
plt.plot(Polars_3, "x:")

plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('Coefficient')
plt.title('2D NACA0012 Polars')
plt.legend()
plt.grid(True)
plt.show()