import {
    buildCsv,
    classIdsFromSelection,
    createToggleMarkup,
    dataUrlFromBase64,
    dominantClass,
    downloadBase64,
    downloadBlob,
    fetchBlob,
    fetchJson,
    formatAreaKm2,
    formatPercent,
    grayscaleArrayFromBase64,
    imageDataFromBase64,
    refreshGlobalHealth,
    renderMaskDataUrl,
    renderOverlayDataUrl,
    renderUncertaintyDataUrl,
    rowsFromStats,
    setImageTarget,
} from "../core.js";

const legend = window.SEGIMBUD_CONFIG.legend || [];

const elements = {
    form: document.getElementById("coordinate-form"),
    latitude: document.getElementById("coordinate-latitude"),
    longitude: document.getElementById("coordinate-longitude"),
    areaKm: document.getElementById("coordinate-area-km"),
    searchDays: document.getElementById("coordinate-search-days"),
    cloudCover: document.getElementById("coordinate-cloud-cover"),
    classToggles: document.getElementById("coordinate-class-toggles"),
    selectAll: document.getElementById("coordinate-select-all"),
    resetButton: document.getElementById("coordinate-reset-button"),
    runButton: document.getElementById("coordinate-run-button"),
    runStatus: document.getElementById("coordinate-run-status"),
    stageImage: document.getElementById("coordinate-stage-image"),
    emptyState: document.getElementById("coordinate-empty-state"),
    previewOriginal: document.getElementById("coordinate-preview-original"),
    previewMask: document.getElementById("coordinate-preview-mask"),
    previewConfidence: document.getElementById("coordinate-preview-confidence"),
    stageLabel: document.getElementById("coordinate-stage-label"),
    viewButtons: [...document.querySelectorAll("#coordinate-view-switcher [data-view]")],
    metricsRibbon: document.getElementById("coordinate-metrics-ribbon"),
    runtimeStack: document.getElementById("coordinate-runtime-stack"),
    sourceStack: document.getElementById("coordinate-source-stack"),
    areaSummary: document.getElementById("coordinate-area-summary"),
    insights: document.getElementById("coordinate-insights"),
    legendWell: document.getElementById("coordinate-legend"),
    statsGrid: document.getElementById("coordinate-stats-grid"),
    downloadReport: document.getElementById("coordinate-download-report"),
    downloadSource: document.getElementById("coordinate-download-source"),
    downloadMask: document.getElementById("coordinate-download-mask"),
    downloadClassMask: document.getElementById("coordinate-download-class-mask"),
    downloadHeatmap: document.getElementById("coordinate-download-heatmap"),
    downloadStats: document.getElementById("coordinate-download-stats"),
};

const state = {
    health: null,
    original: null,
    prediction: null,
    classMask: null,
    confidenceMap: null,
    activeView: "original",
    visibleClasses: new Set(legend.map((entry) => entry.name)),
};

function currentResolution() {
    return Number(state.prediction?.resolution_m_per_px || 0) || 0;
}

function clearState() {
    state.original = null;
    state.prediction = null;
    state.classMask = null;
    state.confidenceMap = null;
}

function dataToBase64(value) {
    if (!value) {
        return "";
    }
    const separatorIndex = value.indexOf(",");
    return separatorIndex >= 0 ? value.slice(separatorIndex + 1) : value;
}

function visibleClassIds() {
    return classIdsFromSelection([...state.visibleClasses], legend);
}

function runtimeCards() {
    const processing = state.prediction?.processing || {};
    return [
        { label: "API status", value: state.health?.ready ? "Ready" : state.health?.status || "Unknown" },
        { label: "Service", value: state.prediction?.source_metadata?.provider || "NASA GIBS" },
        { label: "Processing mode", value: processing.mode || "Idle" },
        { label: "Tile count", value: processing.tile_count || 0 },
    ];
}

function coordinateMetricPairs(rows) {
    if (!state.prediction) {
        return [];
    }

    const metadata = state.prediction.source_metadata || {};
    const confidence = state.prediction.confidence_summary || {};
    const totalArea = rows.reduce((sum, row) => sum + (row.areaSqKm || 0), 0);
    return [
        { label: "Inference", value: `${state.prediction.inference_time} s` },
        { label: "Dominant class", value: dominantClass(state.prediction.stats) },
        { label: "Mean confidence", value: Number(confidence.mean_confidence || 0).toFixed(3) },
        { label: "Acquired", value: metadata.acquired_at ? metadata.acquired_at.slice(0, 10) : "-" },
        { label: "Layer", value: metadata.imagery_layer || metadata.collection || "-" },
        { label: "Scene area", value: currentResolution() > 0 ? `${totalArea.toFixed(6)} km^2` : "Awaiting scale" },
    ];
}

function buildStageViews() {
    if (!state.original) {
        return {};
    }

    const activeMaskIds = visibleClassIds();
    const originalSrc = state.original.src;

    return {
        original: originalSrc,
        mask:
            state.classMask && state.prediction
                ? renderMaskDataUrl(state.classMask, legend, activeMaskIds)
                : state.prediction?.mask_image
                  ? dataUrlFromBase64(state.prediction.mask_image)
                  : originalSrc,
        overlay:
            state.classMask && state.original
                ? renderOverlayDataUrl(state.original, state.classMask, legend, activeMaskIds, 0.55)
                : originalSrc,
        confidence: state.prediction?.heatmap_image ? dataUrlFromBase64(state.prediction.heatmap_image) : originalSrc,
        uncertainty:
            state.original && state.confidenceMap
                ? renderUncertaintyDataUrl(state.original, state.confidenceMap, 0.6)
                : originalSrc,
    };
}

function createCoordinateReportPayload(rows) {
    const views = buildStageViews();
    const metadata = state.prediction?.source_metadata || {};
    return {
        report_title: "SegImBud Coordinate Analysis Report",
        mode_label: "Coordinate analysis",
        summary_metrics: coordinateMetricPairs(rows).slice(0, 4).map((metric) => ({
            label: metric.label,
            value: metric.value,
        })),
        image_panels: [
            {
                title: "Fetched scene",
                image_base64: dataToBase64(state.original?.canvas?.toDataURL("image/png") || ""),
            },
            {
                title: "Operational overlay",
                image_base64: dataToBase64(views.overlay || ""),
            },
            {
                title: "Confidence heatmap",
                image_base64: dataToBase64(views.confidence || ""),
            },
        ],
        insights: state.prediction?.operational_insights || [],
        stats_rows: rows.map((row) => ({
            class: row.class,
            pixels: row.pixels.toLocaleString(),
            percentage: `${row.percentage.toFixed(2)}%`,
            area_sq_km: row.areaSqKm !== null ? row.areaSqKm.toFixed(6) : "-",
        })),
        extra_notes: [
            `Source provider: ${metadata.provider || "NASA GIBS"}.`,
            `Satellite product: ${metadata.satellite || "Daily browse imagery"}.`,
            metadata.imagery_layer ? `Imagery layer: ${metadata.imagery_layer}.` : "Imagery layer unavailable.",
            metadata.item_url ? `Scene URL: ${metadata.item_url}.` : "Scene URL unavailable.",
        ],
    };
}

function renderRuntime() {
    elements.runtimeStack.innerHTML = runtimeCards()
        .map(
            (card) => `
                <article class="runtime-card">
                    <div class="runtime-card__label">${card.label}</div>
                    <div class="runtime-card__value">${card.value}</div>
                </article>
            `,
        )
        .join("");
}

function renderSourceMetadata() {
    if (!state.prediction?.source_metadata) {
        elements.sourceStack.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__value">Fetched scene metadata will appear here after a coordinate run.</div>
            </article>
        `;
        return;
    }

    const metadata = state.prediction.source_metadata;
    const cards = [
        { label: "Provider", value: metadata.provider || "-" },
        { label: "Satellite", value: metadata.satellite || "-" },
        { label: "Layer", value: metadata.imagery_layer || metadata.collection || "-" },
        { label: "Acquired", value: metadata.acquired_at || "-" },
        { label: "Requested scale", value: `${Number(metadata.resolution_m_per_px || 0).toFixed(2)} m/px` },
        { label: "Native scale", value: `${Number(metadata.native_resolution_m_per_px || 0).toFixed(2)} m/px` },
    ];

    elements.sourceStack.innerHTML = cards
        .map(
            (card) => `
                <article class="runtime-card">
                    <div class="runtime-card__label">${card.label}</div>
                    <div class="runtime-card__value">${card.value}</div>
                </article>
            `,
        )
        .join("");
}

function renderAreaSummary(rows) {
    if (!state.prediction) {
        elements.areaSummary.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__value">Run a coordinate fetch to calculate coverage.</div>
            </article>
        `;
        return;
    }

    const totalArea = rows.reduce((sum, row) => sum + (row.areaSqKm || 0), 0);
    const dominantAreaRow = [...rows].sort((a, b) => (b.areaSqKm || 0) - (a.areaSqKm || 0))[0];
    const metadata = state.prediction.source_metadata || {};

    const cards = [
        { label: "Scene area", value: `${totalArea.toFixed(6)} km^2` },
        { label: "Requested width", value: `${Number(metadata.area_km || 0).toFixed(2)} km` },
        {
            label: "Largest class",
            value: dominantAreaRow ? `${dominantAreaRow.class} | ${formatAreaKm2(dominantAreaRow.areaSqKm)}` : "-",
        },
        { label: "Render mode", value: metadata.render_strategy || "-" },
    ];

    elements.areaSummary.innerHTML = cards
        .map(
            (card) => `
                <article class="runtime-card">
                    <div class="runtime-card__label">${card.label}</div>
                    <div class="runtime-card__value">${card.value}</div>
                </article>
            `,
        )
        .join("");
}

function renderInsights() {
    if (!state.prediction?.operational_insights?.length) {
        elements.insights.innerHTML = `
            <article class="insight-card">
                <div class="insight-card__body">Operational observations will appear here after a successful coordinate run.</div>
            </article>
        `;
        return;
    }

    elements.insights.innerHTML = state.prediction.operational_insights
        .map(
            (insight, index) => `
                <article class="insight-card">
                    <div class="runtime-card__label">Insight ${index + 1}</div>
                    <div class="insight-card__body">${insight}</div>
                </article>
            `,
        )
        .join("");
}

function renderLegend(rows) {
    const rowsByName = new Map(rows.map((row) => [row.class, row]));
    elements.legendWell.innerHTML = legend
        .map((entry) => {
            const row = rowsByName.get(entry.name);
            return `
                <article class="legend-item">
                    <span class="legend-item__swatch" style="background: rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]});"></span>
                    <div>
                        <div class="legend-item__title">${entry.name}</div>
                        <div class="legend-item__meta">${row ? formatPercent(row.percentage) : "0.00%"} | ${
                state.visibleClasses.has(entry.name) ? "Visible" : "Hidden"
            }</div>
                    </div>
                </article>
            `;
        })
        .join("");
}

function renderStats(rows) {
    if (!rows.length) {
        elements.statsGrid.innerHTML = `
            <div class="empty-board">Run a coordinate fetch to populate per-class statistics.</div>
        `;
        return;
    }

    elements.statsGrid.innerHTML = rows
        .map(
            (row) => `
                <article class="stat-card">
                    <div class="stat-card__head">
                        <div class="stat-card__title-wrap">
                            <span class="stat-card__swatch" style="background: rgb(${row.color[0]}, ${row.color[1]}, ${row.color[2]});"></span>
                            <span class="stat-card__title">${row.class}</span>
                        </div>
                        <span class="stat-card__value">${formatPercent(row.percentage)}</span>
                    </div>
                    <div class="stat-card__bar"><span class="stat-card__fill" style="width: ${row.percentage}%;"></span></div>
                    <div class="stat-card__meta">Pixels: ${row.pixels.toLocaleString()}</div>
                    <div class="stat-card__meta">Area: ${formatAreaKm2(row.areaSqKm)}</div>
                </article>
            `,
        )
        .join("");
}

function renderMetrics(rows) {
    const metrics = coordinateMetricPairs(rows);
    elements.metricsRibbon.innerHTML = metrics
        .map(
            (metric) => `
                <article class="metric-tile">
                    <div class="metric-tile__label">${metric.label}</div>
                    <div class="metric-tile__value">${metric.value}</div>
                </article>
            `,
        )
        .join("");
}

function setStageStatus(message) {
    elements.runStatus.textContent = message;
}

function renderStage() {
    if (!state.original) {
        setImageTarget(elements.stageImage, "", elements.emptyState);
        elements.stageLabel.textContent = "No scene loaded";
        return;
    }

    const views = buildStageViews();
    const labels = {
        original: "Fetched scene",
        mask: "Filtered class mask",
        overlay: "Operational overlay",
        confidence: "Confidence heatmap",
        uncertainty: "Low-confidence focus",
    };

    setImageTarget(elements.stageImage, views[state.activeView], elements.emptyState);
    elements.stageLabel.textContent = labels[state.activeView];
    setImageTarget(elements.previewOriginal, views.original);
    setImageTarget(elements.previewMask, views.mask);
    setImageTarget(elements.previewConfidence, views.confidence);
}

function renderExports(rows) {
    const hasPrediction = Boolean(state.prediction);
    elements.downloadReport.disabled = !hasPrediction;
    elements.downloadSource.disabled = !hasPrediction || !state.prediction.original_image;
    elements.downloadMask.disabled = !hasPrediction;
    elements.downloadClassMask.disabled = !hasPrediction || !state.prediction.class_mask_image;
    elements.downloadHeatmap.disabled = !hasPrediction || !state.prediction.heatmap_image;
    elements.downloadStats.disabled = !hasPrediction;

    if (!hasPrediction) {
        return;
    }

    elements.downloadReport.onclick = async () => {
        try {
            setStageStatus("Building coordinate PDF report...");
            const blob = await fetchBlob("/reports/pdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(createCoordinateReportPayload(rows)),
            });
            downloadBlob("segimbud-coordinate-report.pdf", blob, "application/pdf");
            setStageStatus("Coordinate report downloaded.");
        } catch (error) {
            setStageStatus(error.message || "Report export failed.");
        }
    };
    elements.downloadSource.onclick = () => downloadBase64("segimbud-coordinate-source.png", state.prediction.original_image);
    elements.downloadMask.onclick = () => downloadBase64("segimbud-coordinate-mask.png", state.prediction.mask_image);
    elements.downloadClassMask.onclick = () => downloadBase64("segimbud-coordinate-class-mask.png", state.prediction.class_mask_image);
    elements.downloadHeatmap.onclick = () => downloadBase64("segimbud-coordinate-heatmap.png", state.prediction.heatmap_image);
    elements.downloadStats.onclick = () =>
        downloadBlob(
            "segimbud-coordinate-stats.csv",
            buildCsv(
                rows.map((row) => ({
                    class: row.class,
                    pixels: row.pixels,
                    percentage: row.percentage.toFixed(2),
                    area_sq_km: row.areaSqKm !== null ? row.areaSqKm.toFixed(6) : "",
                })),
                ["class", "pixels", "percentage", "area_sq_km"],
            ),
            "text/csv",
        );
}

function renderAll() {
    const rows = rowsFromStats(state.prediction?.stats || {}, currentResolution());
    renderRuntime();
    renderSourceMetadata();
    renderAreaSummary(rows);
    renderInsights();
    renderLegend(rows);
    renderStats(rows);
    renderMetrics(rows);
    renderStage();
    renderExports(rows);
}

function syncVisibleClassesFromInputs() {
    state.visibleClasses = new Set(
        [...elements.classToggles.querySelectorAll("input[type='checkbox']:checked")].map((input) => input.value),
    );
    renderAll();
}

function renderClassToggles() {
    elements.classToggles.innerHTML = legend.map((entry) => createToggleMarkup(entry, true)).join("");
    elements.classToggles.addEventListener("change", syncVisibleClassesFromInputs);
    elements.selectAll.addEventListener("click", () => {
        [...elements.classToggles.querySelectorAll("input[type='checkbox']")].forEach((input) => {
            input.checked = true;
        });
        syncVisibleClassesFromInputs();
    });
}

async function prepareOriginal(base64) {
    state.original = await imageDataFromBase64(base64);
    state.original.src = dataUrlFromBase64(base64);
}

async function handleRun(event) {
    event.preventDefault();
    clearState();
    setStageStatus("Searching NASA GIBS and pulling the latest daily scene...");
    elements.runButton.disabled = true;

    try {
        const payload = await fetchJson("/coordinate-predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                latitude: Number.parseFloat(elements.latitude.value || "0"),
                longitude: Number.parseFloat(elements.longitude.value || "0"),
                area_km: Number.parseFloat(elements.areaKm.value || "5"),
                search_days: Number.parseInt(elements.searchDays.value || "21", 10),
                max_cloud_cover: Number.parseFloat(elements.cloudCover.value || "20"),
                output_size_px: 1024,
            }),
        });

        state.prediction = payload;
        await prepareOriginal(payload.original_image);
        if (payload.class_mask_image) {
            state.classMask = await grayscaleArrayFromBase64(payload.class_mask_image);
        }
        if (payload.confidence_map_image) {
            state.confidenceMap = await grayscaleArrayFromBase64(payload.confidence_map_image);
        }

        renderAll();
        setView("overlay");
        setStageStatus("Coordinate analysis complete.");
    } catch (error) {
        clearState();
        renderAll();
        setStageStatus(error.message || "The coordinate request failed.");
    } finally {
        elements.runButton.disabled = false;
    }
}

function resetWorkspace() {
    clearState();
    state.activeView = "original";
    elements.form.reset();
    elements.latitude.value = "23.8103";
    elements.longitude.value = "90.4125";
    elements.areaKm.value = "5.0";
    elements.searchDays.value = "21";
    elements.cloudCover.value = "20";
    [...elements.classToggles.querySelectorAll("input[type='checkbox']")].forEach((input) => {
        input.checked = true;
    });
    state.visibleClasses = new Set(legend.map((entry) => entry.name));
    renderAll();
    setStageStatus("Coordinate workspace reset.");
}

function setView(view) {
    state.activeView = view;
    elements.viewButtons.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.view === view);
    });
    renderStage();
}

async function init() {
    state.health = await refreshGlobalHealth();
    renderClassToggles();
    renderAll();
    elements.form.addEventListener("submit", handleRun);
    elements.resetButton.addEventListener("click", resetWorkspace);
    elements.viewButtons.forEach((button) => {
        button.addEventListener("click", () => setView(button.dataset.view));
    });
}

init();
