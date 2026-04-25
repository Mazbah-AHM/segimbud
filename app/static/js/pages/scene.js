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
    imageDataFromSource,
    refreshGlobalHealth,
    renderMaskDataUrl,
    renderOverlayDataUrl,
    renderUncertaintyDataUrl,
    rowsFromStats,
    setImageTarget,
} from "../core.js";

const legend = window.SEGIMBUD_CONFIG.legend || [];

const elements = {
    form: document.getElementById("scene-form"),
    file: document.getElementById("scene-file"),
    resolution: document.getElementById("scene-resolution"),
    overlayAlpha: document.getElementById("scene-overlay-alpha"),
    confidenceThreshold: document.getElementById("scene-confidence-threshold"),
    classToggles: document.getElementById("scene-class-toggles"),
    selectAll: document.getElementById("scene-select-all"),
    resetButton: document.getElementById("scene-reset-button"),
    runButton: document.getElementById("scene-run-button"),
    runStatus: document.getElementById("scene-run-status"),
    stageImage: document.getElementById("scene-stage-image"),
    emptyState: document.getElementById("scene-empty-state"),
    previewOriginal: document.getElementById("scene-preview-original"),
    previewMask: document.getElementById("scene-preview-mask"),
    previewConfidence: document.getElementById("scene-preview-confidence"),
    stageLabel: document.getElementById("scene-stage-label"),
    viewButtons: [...document.querySelectorAll("#scene-view-switcher [data-view]")],
    metricsRibbon: document.getElementById("scene-metrics-ribbon"),
    runtimeStack: document.getElementById("scene-runtime-stack"),
    areaSummary: document.getElementById("scene-area-summary"),
    insights: document.getElementById("scene-insights"),
    legendWell: document.getElementById("scene-legend"),
    statsGrid: document.getElementById("scene-stats-grid"),
    downloadMask: document.getElementById("scene-download-mask"),
    downloadClassMask: document.getElementById("scene-download-class-mask"),
    downloadHeatmap: document.getElementById("scene-download-heatmap"),
    downloadStats: document.getElementById("scene-download-stats"),
    downloadReport: document.getElementById("scene-download-report"),
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

function clearPredictionState() {
    state.prediction = null;
    state.classMask = null;
    state.confidenceMap = null;
}

function currentResolution() {
    return Number.parseFloat(elements.resolution.value || "0") || 0;
}

function currentOverlayAlpha() {
    return Number.parseFloat(elements.overlayAlpha.value || "0.58") || 0.58;
}

function currentConfidenceThreshold() {
    return Number.parseFloat(elements.confidenceThreshold.value || "0.6") || 0.6;
}

function pixelAreaSqKm() {
    return currentResolution() > 0 ? (currentResolution() ** 2) / 1_000_000 : null;
}

function visibleClassIds() {
    return classIdsFromSelection([...state.visibleClasses], legend);
}

function dataToBase64(value) {
    if (!value) {
        return "";
    }
    const separatorIndex = value.indexOf(",");
    return separatorIndex >= 0 ? value.slice(separatorIndex + 1) : value;
}

function releaseOriginal() {
    if (state.original?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(state.original.src);
    }
}

function runtimeCards() {
    const payload = state.health || {};
    return [
        { label: "API status", value: payload.ready ? "Ready" : payload.status || "Unknown" },
        { label: "Model", value: payload.model_name || "EffKANSeg" },
        { label: "Device", value: String(payload.device || "Unknown").toUpperCase() },
        {
            label: "Image size",
            value: state.original ? `${state.original.width} x ${state.original.height}` : "Not loaded",
        },
    ];
}

function sceneMetricPairs(rows) {
    if (!state.prediction) {
        return [];
    }

    const confidence = state.prediction.confidence_summary || {};
    const totalArea = rows.reduce((sum, row) => sum + (row.areaSqKm || 0), 0);
    const metrics = [
        { label: "Inference", value: `${state.prediction.inference_time} s` },
        { label: "Dominant class", value: dominantClass(state.prediction.stats) },
        { label: "Mean confidence", value: Number(confidence.mean_confidence || 0).toFixed(3) },
        { label: "Low confidence", value: formatPercent(confidence.low_confidence_percentage || 0) },
        {
            label: "Resolution",
            value: currentResolution() > 0 ? `${currentResolution().toFixed(2)} m/px` : "Set resolution",
        },
        {
            label: "Scene area",
            value: currentResolution() > 0 ? `${totalArea.toFixed(6)} km^2` : "Awaiting scale",
        },
    ];
    return metrics;
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
                ? renderOverlayDataUrl(state.original, state.classMask, legend, activeMaskIds, currentOverlayAlpha())
                : originalSrc,
        confidence: state.prediction?.heatmap_image ? dataUrlFromBase64(state.prediction.heatmap_image) : originalSrc,
        uncertainty:
            state.original && state.confidenceMap
                ? renderUncertaintyDataUrl(state.original, state.confidenceMap, currentConfidenceThreshold())
                : originalSrc,
    };
}

function createSceneReportPayload(rows) {
    const stageViews = buildStageViews();
    return {
        report_title: "SegImBud Scene Analysis Report",
        mode_label: "Scene analysis",
        summary_metrics: sceneMetricPairs(rows).slice(0, 4).map((metric) => ({
            label: metric.label,
            value: metric.value,
        })),
        image_panels: [
            {
                title: "Original scene",
                image_base64: dataToBase64(state.original?.canvas?.toDataURL("image/png") || ""),
            },
            {
                title: "Operational overlay",
                image_base64: dataToBase64(stageViews.overlay || ""),
            },
            {
                title: "Confidence heatmap",
                image_base64: dataToBase64(stageViews.confidence || ""),
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
            `Active lens at export: ${state.activeView}.`,
            `Visible classes: ${[...state.visibleClasses].join(", ") || "None"}.`,
            `Low-confidence threshold: ${currentConfidenceThreshold().toFixed(2)}.`,
            currentResolution() > 0
                ? `Resolution used for area estimates: ${currentResolution().toFixed(2)} m/px.`
                : "Area estimates were not requested in this run.",
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

function renderInsights() {
    if (!state.prediction?.operational_insights?.length) {
        elements.insights.innerHTML = `
            <article class="insight-card">
                <div class="insight-card__body">Model observations will appear here after a successful run.</div>
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

function renderAreaSummary(rows) {
    if (!state.prediction) {
        elements.areaSummary.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__value">Run the model to calculate area coverage.</div>
            </article>
        `;
        return;
    }

    if (currentResolution() <= 0) {
        elements.areaSummary.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__label">Scale required</div>
                <div class="runtime-card__value">Enter meters per pixel to calculate area.</div>
            </article>
        `;
        return;
    }

    const totalArea = rows.reduce((sum, row) => sum + (row.areaSqKm || 0), 0);
    const mappedArea = rows
        .filter((row) => row.class !== "Background")
        .reduce((sum, row) => sum + (row.areaSqKm || 0), 0);
    const dominantAreaRow = [...rows].sort((a, b) => (b.areaSqKm || 0) - (a.areaSqKm || 0))[0];

    const cards = [
        { label: "Resolution", value: `${currentResolution().toFixed(2)} m/px` },
        { label: "Scene area", value: `${totalArea.toFixed(6)} km^2` },
        { label: "Mapped area", value: `${mappedArea.toFixed(6)} km^2` },
        {
            label: "Largest class",
            value: dominantAreaRow ? `${dominantAreaRow.class} | ${formatAreaKm2(dominantAreaRow.areaSqKm)}` : "-",
        },
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
            <div class="empty-board">Run the model to populate per-class statistics for this scene.</div>
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
    const metrics = sceneMetricPairs(rows);
    if (!metrics.length) {
        elements.metricsRibbon.innerHTML = "";
        return;
    }
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

function setView(view) {
    state.activeView = view;
    elements.viewButtons.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.view === view);
    });
    renderStage();
}

function renderStage() {
    if (!state.original) {
        setImageTarget(elements.stageImage, "", elements.emptyState);
        elements.stageLabel.textContent = "No scene loaded";
        return;
    }

    const views = buildStageViews();
    const labels = {
        original: "Original view",
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
    elements.downloadMask.disabled = !hasPrediction;
    elements.downloadClassMask.disabled = !hasPrediction || !state.prediction.class_mask_image;
    elements.downloadHeatmap.disabled = !hasPrediction || !state.prediction.heatmap_image;
    elements.downloadStats.disabled = !hasPrediction;
    elements.downloadReport.disabled = !hasPrediction;

    if (!hasPrediction) {
        return;
    }

    elements.downloadMask.onclick = () => downloadBase64("segimbud-mask.png", state.prediction.mask_image);
    elements.downloadClassMask.onclick = () => downloadBase64("segimbud-class-mask.png", state.prediction.class_mask_image);
    elements.downloadHeatmap.onclick = () => downloadBase64("segimbud-heatmap.png", state.prediction.heatmap_image);
    elements.downloadStats.onclick = () =>
        downloadBlob(
            "segimbud-stats.csv",
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
    elements.downloadReport.onclick = async () => {
        try {
            setStageStatus("Building PDF report...");
            const blob = await fetchBlob("/reports/pdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(createSceneReportPayload(rows)),
            });
            downloadBlob("segimbud-scene-report.pdf", blob, "application/pdf");
            setStageStatus("Scene report downloaded.");
        } catch (error) {
            setStageStatus(error.message || "Report export failed.");
        }
    };
}

function renderAll() {
    const rows = rowsFromStats(state.prediction?.stats || {}, currentResolution());
    renderRuntime();
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

async function prepareOriginal(file) {
    releaseOriginal();
    const sourceUrl = URL.createObjectURL(file);
    state.original = {
        src: sourceUrl,
        ...(await imageDataFromSource(sourceUrl)),
    };
}

async function handleFileSelection() {
    const file = elements.file.files?.[0];
    clearPredictionState();

    if (!file) {
        releaseOriginal();
        state.original = null;
        renderAll();
        setStageStatus("Waiting for scene upload.");
        return;
    }

    try {
        await prepareOriginal(file);
        renderAll();
        setView("original");
        setStageStatus(`Loaded ${file.name}. Ready to run inference.`);
    } catch (error) {
        releaseOriginal();
        state.original = null;
        renderAll();
        setStageStatus(error.message || "The selected image could not be loaded.");
    }
}

async function handleRun(event) {
    event.preventDefault();
    const file = elements.file.files?.[0];
    if (!file) {
        setStageStatus("Choose a scene file before running the model.");
        return;
    }

    clearPredictionState();
    setStageStatus("Running scene analysis...");
    elements.runButton.disabled = true;

    try {
        await prepareOriginal(file);
        renderAll();
        setView("original");
        const formData = new FormData();
        formData.append("file", file);
        if (currentResolution() > 0) {
            formData.append("resolution_m_per_px", String(currentResolution()));
        }
        const payload = await fetchJson("/predict", {
            method: "POST",
            body: formData,
        });

        state.prediction = payload;
        if (payload.class_mask_image) {
            state.classMask = await grayscaleArrayFromBase64(payload.class_mask_image);
        }
        if (payload.confidence_map_image) {
            state.confidenceMap = await grayscaleArrayFromBase64(payload.confidence_map_image);
        }

        renderAll();
        setView("overlay");
        setStageStatus("Scene analysis complete.");
    } catch (error) {
        renderAll();
        setStageStatus(error.message || "The analysis request failed.");
    } finally {
        elements.runButton.disabled = false;
    }
}

function resetScene() {
    releaseOriginal();
    state.original = null;
    clearPredictionState();
    state.activeView = "original";
    elements.form.reset();
    [...elements.classToggles.querySelectorAll("input[type='checkbox']")].forEach((input) => {
        input.checked = true;
    });
    state.visibleClasses = new Set(legend.map((entry) => entry.name));
    renderAll();
    setStageStatus("Scene workspace reset.");
}

async function init() {
    state.health = await refreshGlobalHealth();
    renderClassToggles();
    renderAll();
    elements.form.addEventListener("submit", handleRun);
    elements.file.addEventListener("change", handleFileSelection);
    elements.resetButton.addEventListener("click", resetScene);
    elements.overlayAlpha.addEventListener("input", renderStage);
    elements.confidenceThreshold.addEventListener("input", renderStage);
    elements.resolution.addEventListener("input", renderAll);
    elements.viewButtons.forEach((button) => {
        button.addEventListener("click", () => setView(button.dataset.view));
    });
}

init();
