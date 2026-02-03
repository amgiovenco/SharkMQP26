import Plotly from 'react-plotly.js';

const MeltingCurveChart = ({ frequencies, signal, predictedSpecies }) => {
    // Handle missing data
    if (!frequencies || !signal || frequencies.length === 0) {
        return (
            <div className="w-full h-48 flex items-center justify-center bg-gray-50 rounded p-4">
                <p className="text-sm text-gray-500">No curve data available</p>
            </div>
        );
    }

    const trace = {
        x: frequencies,
        y: signal,
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#2E86AB',
            width: 2,
        },
        hovertemplate: '<b>%{x:.2f}°C</b><br>Fluorescence: %{y:.4f}<extra></extra>',
    };

    const layout = {
        title: {
            text: `Melting Curve - ${predictedSpecies}`,
            font: { size: 12 },
        },
        xaxis: {
            title: 'Temperature (°C)',
            showgrid: true,
            gridwidth: 1,
            gridcolor: '#e5e7eb',
        },
        yaxis: {
            title: 'Fluorescence',
            showgrid: true,
            gridwidth: 1,
            gridcolor: '#e5e7eb',
        },
        margin: { l: 40, r: 20, t: 40, b: 40 },
        hovermode: 'x unified',
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
    };

    const config = {
        responsive: true,
        displayModeBar: false,
        staticPlot: false,
    };

    return (
        <div className="w-full h-48">
            <Plotly
                data={[trace]}
                layout={layout}
                config={config}
                style={{ width: '100%', height: '100%' }}
            />
        </div>
    );
};

export default MeltingCurveChart;
