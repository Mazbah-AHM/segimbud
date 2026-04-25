import {
    buildCsv,
    classIdsFromSelection,
    createToggleMarkup,
    dataUrlFromBase64,
    dominantClass,
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
    rowsFromStats,
    setImageTarget,
} from "../core.js";

const legend = window.SEGIMBUD_CONFIG.legend || [];

const elements = {
    form: document.getElementById("batch-form"),
    files: document.getElementById("batch-files"),
    resolution: document.getElementById("batch-resolution"),
    tileSize: document.getElementById("batch-tile-size"),
    tileOverlap: document.getElementById("batch-tile-overlap"),
    forceTiled: document.getElementById("batch-force-tiled"),
    classToggles: document.getElementById("batch-class-toggles"),
    selectAll: document.getElementById("batch-select-all"),
    resetButton: document.getElementById("batch-reset-button"),
    runButton: document.getElementById("batch-run-button"),
    runStatus: document.getElementById("batch-run-status"),
    stageImage: document.getElementById("batch-stage-image"),
    emptyState: document.getElementById("batch-empty-state"),
    previewOriginal: document.getElementById("batch-preview-original"),
    previewMask: document.getElementById("batch-preview-mask"),
    previewConfidence: document.getElementById("batch-preview-confidence"),
    stageLabel: document.getElementById("batch-stage-label"),
    viewButtons: [...document.querySelectorAll("#batch-view-switcher [data-view]")],
    metricsRibbon: document.getElementById("batch-metrics-ribbon"),
    summaryStack: document.getElementById("batch-summary-stack"),
    processingStack: document.getElementById("batch-processing-stack"),
    resultsList: document.getElementById("batch-results-list"),
    statsGrid: document.getElementById("batch-stats-grid"),
    downloadReport: document.getElementById("batch-download-report"),
    downloadMask: document.getElementById("batch-download-mask"),
    downloadStats: document.getElementById("batch-download-stats"),
    downloadSummary: document.getElementById("batch-download-summary"),
};

const state = {
    health: null,
    summary: null,
    results: [],
    selectedIndex: -1,
    activeView: "original",
    visibleClasses: new Set(legend.map((entry) => entry.name)),
};

function currentResolution() {
    return Number.parseFloat(elements.resolution.value || "0") || 0;
}

function visibleClassIds() {
    return classIdsFromSelection([...state.visibleClasses], legend);
}

function selectedResult() {
    const candidate = state.results[state.selectedIndex];
    return candidate?.ok ? candidate : null;
}

function setStageStatus(message) {
    elements.runStatus.textContent = message;
}

function buildStageViews(result) {
    if (!result?.original) {
        return {};
    }

    const activeMaskIds = visibleClassIds();
    const originalSrc = result.original.src;

    return {
        original: originalSrc,
        mask:
            result.classMask && result.prediction
                ? renderMaskDataUrl(result.classMask, legend, activeMaskIds)
                : result.prediction?.mask_image
                  ? dataUrlFromBase64(result.prediction.mask_image)
                  : originalSrc,
        overlay:
            result.classMask && result.original
                ? renderOverlayDataUrl(result.original, result.classMask, legend, activeMaskIds, 0.55)
                : originalSrc,
        confidence: result.prediction?.heatmap_image ? dataUrlFromBase64(result.prediction.heatmap_image) : originalSrc,
    };
}

function selectedRows() {
    const result = selectedResult();
    return rowsFromStats(result?.prediction?.stats || {}, currentResolution());
}

function batchMetricPairs() {
    const result = selectedResult();
    if (!result) {
        return [];
    }

    const processing = result.prediction.processing || {};
    const confidence = result.prediction.confidence_summary || {};
    return [
        { label: "Inference", value: `${result.prediction.inference_time} s` },
        { label: "Dominant class", value: dominantClass(result.prediction.stats) },
        { label: "Mean confidence", value: Number(confidence.mean_confidence || 0).toFixed(3) },
        { label: "Mode", value: processing.mode || "-" },
        { label: "Tiles", value: processing.tile_count || 0 },
        { label: "Tile size", value: `${processing.tile_size || 0}px` },
    ];
}

function createSelectedReportPayload(rows) {
    const result = selectedResult();
    const views = buildStageViews(result);
    if (!result) {
        return null;
    }

    return {
        report_title: `SegImBud Batch Report - ${result.filename}`,
        mode_label: "Batch processing",
        summary_metrics: batchMetricPairs().slice(0, 4).map((metric) => ({
            label: metric.label,
            value: metric.value,
        })),
        image_panels: [
            {
                title: "Original scene",
                image_base64: result.original.canvas.toDataURL("image/png").split(",")[1],
            },
            {
                title: "Operational overlay",
                image_base64: (views.overlay || "").split(",")[1] || "",
            },
            {
                title: "Confidence heatmap",
                image_base64: (views.confidence || "").split(",")[1] || "",
            },
        ],
        insights: result.prediction.operational_insights || [],
        stats_rows: rows.map((row) => ({
            class: row.class,
            pixels: row.pixels.toLocaleString(),
            percentage: `${row.percentage.toFixed(2)}%`,
            area_sq_km: row.areaSqKm !== null ? row.areaSqKm.toFixed(6) : "-",
        })),
        extra_notes: [
            `Filename: ${result.filename}.`,
            `Processing mode: ${result.prediction.processing?.mode || "single"}.`,
            `Tile count: ${result.prediction.processing?.tile_count || 1}.`,
        ],
    };
}

function renderSummary() {
    if (!state.summary) {
        elements.summaryStack.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__value">Batch summary will appear after a run.</div>
            </article>
        `;
        return;
    }

    const cards = [
        { label: "Requested", value: state.summary.requested_count || 0 },
        { label: "Completed", value: state.summary.successful_count || 0 },
        { label: "Failed", value: state.summary.failed_count || 0 },
        { label: "Tiles total", value: state.summary.total_tile_count || 0 },
        { label: "Inference total", value: `${Number(state.summary.total_inference_time || 0).toFixed(3)} s` },
    ];

    elements.summaryStack.innerHTML = cards
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

function renderProcessingDetails() {
    const result = selectedResult();
    if (!result) {
        elements.processingStack.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__value">Select a successful result to inspect its processing details.</div>
            </article>
        `;
        return;
    }

    const processing = result.prediction.processing || {};
    const cards = [
        { label: "Filename", value: result.filename },
        { label: "Image size", value: `${result.original.width} x ${result.original.height}` },
        { label: "Processing mode", value: processing.mode || "-" },
        { label: "Tile overlap", value: `${processing.tile_overlap || 0}px` },
        { label: "Stride", value: `${processing.stride || 0}px` },
    ];

    elements.processingStack.innerHTML = cards
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

function renderResultsList() {
    if (!state.results.length) {
        elements.resultsList.innerHTML = `
            <div class="empty-board">Run a batch job to populate the processed queue.</div>
        `;
        return;
    }

    elements.resultsList.innerHTML = state.results
        .map((result, index) => {
            const dominant = result.ok ? dominantClass(result.prediction.stats) : "Error";
            const mode = result.ok ? result.prediction.processing?.mode || "-" : "failed";
            return `
                <article class="queue-card ${state.selectedIndex === index ? "is-active" : ""}" data-index="${index}">
                    <div class="queue-card__head">
                        <div class="queue-card__title">${result.filename}</div>
                        <div class="pill ${result.ok ? "is-good" : "is-bad"}">${result.ok ? "Ready" : "Failed"}</div>
                    </div>
                    <div class="queue-card__meta">${result.ok ? `Dominant class: ${dominant}` : result.error}</div>
                    <div class="queue-card__meta">Mode: ${mode}</div>
                </article>
            `;
        })
        .join("");

    [...elements.resultsList.querySelectorAll("[data-index]")].forEach((card) => {
        card.addEventListener("click", () => {
            state.selectedIndex = Number(card.dataset.index);
            renderAll();
        });
    });
}

function renderStatsGrid(rows) {
    if (!rows.length) {
        elements.statsGrid.innerHTML = `
            <div class="empty-board">Select a successful result to inspect its class distribution.</div>
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

function renderMetrics() {
    const metrics = batchMetricPairs();
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

function renderStage() {
    const result = selectedResult();
    if (!result) {
        setImageTarget(elements.stageImage, "", elements.emptyState);
        elements.stageLabel.textContent = "None";
        return;
    }

    const views = buildStageViews(result);
    const labels = {
        original: result.filename,
        mask: `${result.filename} | mask`,
        overlay: `${result.filename} | overlay`,
        confidence: `${result.filename} | confidence`,
    };

    setImageTarget(elements.stageImage, views[state.activeView], elements.emptyState);
    elements.stageLabel.textContent = labels[state.activeView] || result.filename;
    setImageTarget(elements.previewOriginal, views.original);
    setImageTarget(elements.previewMask, views.mask);
    setImageTarget(elements.previewConfidence, views.confidence);
}

function renderExports(rows) {
    const result = selectedResult();
    const ready = Boolean(result);
    elements.downloadReport.disabled = !ready;
    elements.downloadMask.disabled = !ready;
    elements.downloadStats.disabled = !ready;
    elements.downloadSummary.disabled = !state.results.length;

    if (ready) {
        elements.downloadReport.onclick = async () => {
            try {
                setStageStatus("Building selected batch PDF report...");
                const blob = await fetchBlob("/reports/pdf", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(createSelectedReportPayload(rows)),
                });
                downloadBlob(`segimbud-batch-${result.filename}-report.pdf`, blob, "application/pdf");
                setStageStatus("Selected batch report downloaded.");
            } catch (error) {
                setStageStatus(error.message || "Report export failed.");
            }
        };
        elements.downloadMask.onclick = () => {
            downloadBlob(
                `segimbud-batch-${result.filename}-mask.png`,
                Uint8Array.from(atob(result.prediction.mask_image), (char) => char.charCodeAt(0)),
                "image/png",
            );
        };
        elements.downloadStats.onclick = () =>
            downloadBlob(
                `segimbud-batch-${result.filename}-stats.csv`,
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

    if (state.results.length) {
        elements.downloadSummary.onclick = () =>
            downloadBlob(
                "segimbud-batch-summary.csv",
                buildCsv(
                    state.results.map((result) => ({
                        filename: result.filename,
                        status: result.ok ? "ready" : "failed",
                        dominant_class: result.ok ? dominantClass(result.prediction.stats) : "",
                        inference_time_s: result.ok ? result.prediction.inference_time : "",
                        tile_count: result.ok ? result.prediction.processing?.tile_count || 0 : "",
                        error: result.ok ? "" : result.error,
                    })),
                    ["filename", "status", "dominant_class", "inference_time_s", "tile_count", "error"],
                ),
                "text/csv",
            );
    }
}

function renderAll() {
    const rows = selectedRows();
    renderSummary();
    renderProcessingDetails();
    renderResultsList();
    renderStatsGrid(rows);
    renderMetrics();
    renderStage();
    renderExports(rows);
}

function setView(view) {
    state.activeView = view;
    elements.viewButtons.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.view === view);
    });
    renderStage();
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

async function buildOriginal(file) {
    const sourceUrl = URL.createObjectURL(file);
    return {
        src: sourceUrl,
        ...(await imageDataFromSource(sourceUrl)),
    };
}

async function handleRun(event) {
    event.preventDefault();
    const files = [...(elements.files.files || [])];
    if (!files.length) {
        setStageStatus("Choose at least one image before running a batch job.");
        return;
    }

    setStageStatus("Running tiled batch inference...");
    elements.runButton.disabled = true;
    state.results = [];
    state.summary = null;
    state.selectedIndex = -1;
    renderAll();

    try {
        const originals = await Promise.all(files.map((file) => buildOriginal(file)));
        const formData = new FormData();
        files.forEach((file) => formData.append("files", file));
        if (currentResolution() > 0) {
            formData.append("resolution_m_per_px", String(currentResolution()));
        }
        formData.append("tile_size", String(Number.parseInt(elements.tileSize.value || "1024", 10)));
        formData.append("tile_overlap", String(Number.parseInt(elements.tileOverlap.value || "160", 10)));
        formData.append("force_tiled", elements.forceTiled.checked ? "true" : "false");

        const payload = await fetchJson("/batch-predict", {
            method: "POST",
            body: formData,
        });

        state.summary = payload.summary || null;
        state.results = await Promise.all(
            (payload.results || []).map(async (result, index) => {
                const combined = {
                    ...result,
                    original: originals[index],
                };
                if (result.ok) {
                    combined.classMask = await grayscaleArrayFromBase64(result.prediction.class_mask_image);
                    combined.confidenceMap = await grayscaleArrayFromBase64(result.prediction.confidence_map_image);
                }
                return combined;
            }),
        );
        state.selectedIndex = state.results.findIndex((result) => result.ok);
        renderAll();
        if (state.selectedIndex >= 0) {
            setView("overlay");
        }
        setStageStatus("Batch processing complete.");
    } catch (error) {
        state.results = [];
        state.summary = null;
        state.selectedIndex = -1;
        renderAll();
        setStageStatus(error.message || "The batch request failed.");
    } finally {
        elements.runButton.disabled = false;
    }
}

function resetWorkspace() {
    state.summary = null;
    state.results.forEach((result) => {
        if (result.original?.src?.startsWith("blob:")) {
            URL.revokeObjectURL(result.original.src);
        }
    });
    state.results = [];
    state.selectedIndex = -1;
    state.activeView = "original";
    elements.form.reset();
    elements.resolution.value = "0.5";
    elements.tileSize.value = "1024";
    elements.tileOverlap.value = "160";
    elements.forceTiled.checked = true;
    [...elements.classToggles.querySelectorAll("input[type='checkbox']")].forEach((input) => {
        input.checked = true;
    });
    state.visibleClasses = new Set(legend.map((entry) => entry.name));
    renderAll();
    setStageStatus("Batch workspace reset.");
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
