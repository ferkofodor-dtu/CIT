import pycountry_convert as pc
import pandas as pd
import matplotlib.pyplot as plt

AGGREGATE_ENTITIES = {
    "World",
    "Africa",
    "Africa (UN)",
    "Americas (UN)",
    "Asia",
    "Asia (UN)",
    "Europe",
    "Europe (UN)",
    "High-income countries",
    "Low-income countries",
    "Lower-middle-income countries",
    "Upper-middle-income countries",
    "Land-locked developing countries (LLDC)",
    "Least developed countries",
    "Less developed regions, excluding least developed countries",
    "Latin America and the Caribbean (UN)",
    "North America",
    "Northern America (UN)",
    "Oceania",
    "Oceania (UN)",
    "South America",
    "Less developed regions",
    "Less developed regions, excluding China",
    "More developed regions",
    "Small island developing states (SIDS)",
    "East Asia and Pacific (WB)",
    "European Union (27)",
    "European Union (28)",
    "Latin America and Caribbean (WB)",
    "Middle East, North Africa, Afghanistan and Pakistan (WB)",
    "North America (WB)",
    "South Asia (WB)",
    "Sub-Saharan Africa (WB)",
    "Europe and Central Asia (WB)"
    
}


def remove_aggregate_rows(df, entity_col="country"):
    """
    Remove aggregate regional/income-group rows from an OWID-style dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    entity_col : str
        Column containing country/entity names.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe containing country-level rows only.
    """
    cleaned_df = df[~df[entity_col].isin(AGGREGATE_ENTITIES)].copy()
    return cleaned_df


def add_continent_column(df, code_col="country_code"):
    """
    Add a continent column based on ISO-3 country codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    code_col : str
        Column containing ISO-3 country codes

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'continent' column
    """

    def code_to_continent(code):
        try:
            country_alpha2 = pc.country_alpha3_to_country_alpha2(code)
            continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            continent_map = {
                "AF": "Africa",
                "AS": "Asia",
                "EU": "Europe",
                "NA": "North America",
                "SA": "South America",
                "OC": "Oceania"
            }
            return continent_map.get(continent_code, None)
        except Exception:
            return None

    df = df.copy()
    df["continent"] = df[code_col].apply(code_to_continent)

    # add manual mappings for missing codes
    manual_mappings = {
        'TLS': 'Asia',  # East Timor
        'XK': 'Europe',  # Kosovo
        'SXM': 'North America',  # Sint Maarten (Dutch part)
        'VAT': 'Europe',  # Vatican
        'ESH': 'Africa'  # Western Sahara
    }
    df["continent"] = df.apply(lambda row: manual_mappings.get(row[code_col], row["continent"]), axis=1)

    return df


def load_clean_data(path):
    df = pd.read_csv(path)
    df = remove_aggregate_rows(df)
    df = add_continent_column(df)
    return df



def plot_slope_chart(
    df,
    value_col,
    group_col,
    start_year,
    end_year,
    top_k=10,
    year_col="year",
    figsize=(10, 8),
    title=None,
    sort_by="end",
    aggregate_func="sum",
    annotate=True,
    ax=None
):
    
    valid_sort = {"start", "end"}
    if sort_by not in valid_sort:
        raise ValueError(f"sort_by must be one of {valid_sort}")

    # Keep only relevant years
    filtered = df[df[year_col].isin([start_year, end_year])].copy()

    # Aggregate
    agg_df = (
        filtered.groupby([group_col, year_col], as_index=False)[value_col]
        .agg(aggregate_func)
    )

    # Pivot to wide format
    plot_df = (
        agg_df.pivot(index=group_col, columns=year_col, values=value_col)
        .rename(columns={start_year: "start_value", end_year: "end_value"})
        .dropna(subset=["start_value", "end_value"])
        .reset_index()
    )

    # Select top_k
    sort_col = "start_value" if sort_by == "start" else "end_value"
    plot_df = (
        plot_df.sort_values(sort_col, ascending=False)
        .head(top_k)
        .sort_values("end_value", ascending=True)
        .reset_index(drop=True)
    )

    # Plot
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    x0, x1 = 0, 1

    for _, row in plot_df.iterrows():
        color = "tab:blue" if row["end_value"] >= row["start_value"] else "tab:red"

        ax.plot(
            [x0, x1],
            [row["start_value"], row["end_value"]],
            color=color,
            linewidth=2
        )

        if annotate:
            ax.text(
                x0 - 0.02,
                row["start_value"],
                f"{row[group_col]} {row['start_value']:,.2f}",
                ha="right",
                va="center"
            )
            ax.text(
                x1 + 0.02,
                row["end_value"],
                f"{row[group_col]} {row['end_value']:,.2f}",
                ha="left",
                va="center",
                color=color
            )

    # Vertical guides
    ax.axvline(x=x0, color="lightgray", linewidth=1)
    ax.axvline(x=x1, color="lightgray", linewidth=1)

    # Axes formatting
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([start_year, end_year], fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(-0.25, 1.25)

    if title is None:
        title = f"{value_col.replace('_', ' ').title()}: {start_year} vs {end_year}"

    ax.set_title(title)
    return plot_df