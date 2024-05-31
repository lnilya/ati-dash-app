from src.__libs import pyutil
from src.datautil import PlotProps, datautil
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go


def render2D(grA, vars, col: str = None, showPlot=True, highlightSpp=None):
    grA = grA.loc[grA.Type == "Gr"].dropna()
    vars = PlotProps.renameBioClimToClearTextList(vars, useShortNames=False)
    PlotProps.renameBioClimToClearText(grA, useShortNames=False)

    # number of species goint up and down along variables
    incLoc0 = 100 * (np.count_nonzero(grA[vars[0]] > 0) / len(grA))
    incLoc1 = 100 * (np.count_nonzero(grA[vars[1]] > 0) / len(grA))

    # load species info
    covers, si = datautil.getSpeciesInfo(grA, useCols=["Species","Cover","GrowthForm","Family","BioStatus","ThreatenedStatus","NumObs"])
    elx, ely = pyutil.get2DMaxLikelihoodCovarianceEllipse(grA, vars, covers["Cover"])
    elx2, ely2 = pyutil.get2DMaxLikelihoodCovarianceEllipse(grA, vars, covers["Cover"], numStd=2)
    wmx, wmy = np.average(covers[vars[0]], weights=covers["Cover"]), np.average(covers[vars[1]],
                                                                                weights=covers["Cover"])

    rng = [grA[vars[0]].min(), grA[vars[0]].max(), grA[vars[1]].min(), grA[vars[1]].max()]
    dx = (rng[1] - rng[0]) * 0.05
    dy = (rng[3] - rng[2]) * 0.05
    rng = [rng[0] - dx, rng[1] + dx, rng[2] - dy, rng[3] + dy]
    # run a weighted KDE for display of the cover
    rx = np.linspace(rng[0], rng[1], 200)
    ry = np.linspace(rng[2], rng[3], 200)
    X, Y = np.meshgrid(rx, ry)
    positions = np.vstack([X.ravel(), Y.ravel()])

    kde = stats.gaussian_kde([grA[vars[0]], grA[vars[1]]], weights=covers["Cover"])
    z = kde(positions)
    z = np.reshape(z.T, X.shape)

    # Plot the heatmap

    fig = go.Figure()
    fig.add_heatmap(x=rx, y=ry, z=z, colorscale=px.colors.sequential.haline, colorbar=None, showscale=False,

                    hoverinfo="skip")

    # plot mean and covariance as ellipses
    fig.add_trace(
        go.Scatter(x=elx, y=ely, mode='lines', line=dict(color="red", width=2, dash="dot"), name="1σ"))
    fig.add_trace(
        go.Scatter(x=elx2, y=ely2, mode='lines', line=dict(color="red", width=1, dash="dot"), name="2σ"))

    # plot the 0 lines
    fig.add_shape(type="line", x0=0, y0=ry.min(), x1=0, y1=ry.max(), line=dict(color="white", width=1))
    fig.add_shape(type="line", x0=rx.min(), y0=0, x1=rx.max(), y1=0, line=dict(color="white", width=1))

    # plot the species and the mean
    # markers need to be either cross or cross-open depending on pValues

    cols = px.colors.qualitative.Light24
    nc = len(cols)
    if col is not None:
        grC =  covers.groupby(col)
        for i,( col, gr) in enumerate(grC):
            fig.add_scatter(x=gr[vars[0]], y=gr[vars[1]], mode='markers',
                            marker=dict(size=8, color=cols[i%nc],
                                        symbol="cross", line=dict(width=.5, color="black")), name=col, hovertext=gr["Species"])
    else:
        fig.add_scatter(x=grA[vars[0]], y=grA[vars[1]], mode='markers',
                        marker=dict(size=8, color="white", symbol="cross", line=dict(width=.5, color="black")), name="Species", hovertext=grA["Species"])

    fig.add_scatter(x=[wmx], y=[wmy], mode='markers', marker=dict(size=10, color="red", symbol="x"),
                    name="Average shift by cover")

    fig.update_xaxes(range=[rx.min(), rx.max()],
                     title_text=f"Δ {vars[0]} (↑ {incLoc0:.1f} % / ↓ {(100 - incLoc0):.1f} %)")
    fig.update_yaxes(range=[ry.min(), ry.max()],
                     title_text=f"Δ {vars[1]} (↑ {incLoc1:.1f} % / ↓ {(100 - incLoc1):.1f} %)")

    fig.update_layout(coloraxis_showscale=False)
    # remove legend and colorscale
    if showPlot:
        fig.show()

    return fig
