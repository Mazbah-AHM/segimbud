import {
    buildCsv,
    downloadBlob,
    fetchBlob,
    fetchJson,
    formatAreaKm2,
    formatPercent,
    grayscaleArrayFromBase64,
    imageDataFromSource,
    refreshGlobalHealth,
    rowsFromMask,
} from "../core.js";

const legend = window.SEGIMBUD_CONFIG.legend || [];
const legendById = new Map(legend.map((entry) => [Number(entry.class_id), entry]));

const elements = {
    form: document.getElementById("review-form"),
    file: document.getElementById("review-file"),
    resolution: document.getElementById("review-resolution"),
    statusSelect: document.getElementById("review-status-select"),
    reviewerName: document.getElementById("reviewer-name"),
    brushClass: document.getElementById("review-brush-class"),
    brushSize: document.getElementById("review-brush-size"),
    zoom: document.getElementById("review-zoom"),
    confidenceThreshold: document.getElementById("review-confidence-threshold"),
    notes: document.getElementById("review-notes"),
    runButton: document.getElementById("review-run-button"),
    resetButton: document.getElementById("review-reset-button"),
    runStatus: document.getElementById("review-run-status"),
    canvas: document.getElementById("review-stage-canvas"),
    emptyState: document.getElementById("review-empty-state"),
    viewButtons: [...document.querySelectorAll("#review-view-switcher [data-view]")],
    metricsRibbon: document.getElementById("review-metrics-ribbon"),
    qaSummary: document.getElementById("review-qa-summary"),
    touchedClasses: document.getElementById("review-touched-classes"),
    legendWell: document.getElementById("review-legend"),
    statsGrid: document.getElementById("review-stats-grid"),
    downloadReport: document.getElementById("review-download-report"),
    downloadMask: document.getElementById("review-download-mask"),
    downloadStats: document.getElementById("review-download-stats"),
    downloadQa: document.getElementById("review-download-qa"),
};

const state = {
    health: null,
    selectedFile: null,
    original: null,
    prediction: null,
    originalMask: null,
    classMask: null,
    confidenceMap: null,
    activeView: "overlay",
    isPainting: false,
};

function currentResolution() {
    return Number.parseFloat(elements.resolution.value || "0") || 0;
}

function currentBrushClassId() {
    return Number.parseInt(elements.brushClass.value || "0", 10);
}

function currentBrushSize() {
    return Number.parseInt(elements.brushSize.value || "18", 10);
}

function currentZoom() {
    return Number.parseFloat(elements.zoom.value || "1") || 1;
}

function currentConfidenceThreshold() {
    return Number.parseFloat(elements.confidenceThreshold.value || "0.6") || 0.6;
}

function setStageStatus(message) {
    elements.runStatus.textContent = message;
}

function clearReviewState() {
    if (state.original?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(state.original.src);
    }
    state.original = null;
    state.prediction = null;
    state.originalMask = null;
    state.classMask = null;
    state.confidenceMap = null;
}

function currentFile() {
    return state.selectedFile || elements.file.files?.[0] || null;
}

function cloneMask(mask) {
    return {
        width: mask.width,
        height: mask.height,
        data: new Uint8ClampedArray(mask.data),
    };
}

function showCanvasState() {
    const hasScene = Boolean(state.original && state.classMask);
    elements.emptyState.classList.toggle("is-hidden", hasScene);
    elements.canvas.classList.toggle("is-visible", hasScene);
}

function paintCircle(clientX, clientY) {
    if (!state.classMask || !state.original) {
        return;
    }

    const rect = elements.canvas.getBoundingClientRect();
    const x = Math.round(((clientX - rect.left) / rect.width) * state.classMask.width);
    const y = Math.round(((clientY - rect.top) / rect.height) * state.classMask.height);
    const radius = currentBrushSize();
    const classId = currentBrushClassId();

    const left = Math.max(0, x - radius);
    const right = Math.min(state.classMask.width - 1, x + radius);
    const top = Math.max(0, y - radius);
    const bottom = Math.min(state.classMask.height - 1, y + radius);

    for (let row = top; row <= bottom; row += 1) {
        for (let column = left; column <= right; column += 1) {
            const deltaX = column - x;
            const deltaY = row - y;
            if (deltaX * deltaX + deltaY * deltaY > radius * radius) {
                continue;
            }
            state.classMask.data[row * state.classMask.width + column] = classId;
        }
    }
}

function computeReviewRows() {
    return rowsFromMask(state.classMask, legend, currentResolution());
}

function computeQaSummary() {
    if (!state.classMask || !state.originalMask) {
        return {
            editedPixels: 0,
            editedShare: 0,
            touchedClasses: [],
            lowConfidenceEditedPixels: 0,
            lowConfidenceEditedShare: 0,
        };
    }

    let editedPixels = 0;
    let lowConfidenceEditedPixels = 0;
    const touchedClassIds = new Set();
    for (let index = 0; index < state.classMask.data.length; index += 1) {
        if (state.classMask.data[index] !== state.originalMask.data[index]) {
            editedPixels += 1;
            touchedClassIds.add(Number(state.classMask.data[index]));
            if (state.confidenceMap && state.confidenceMap.data[index] / 255 < currentConfidenceThreshold()) {
                lowConfidenceEditedPixels += 1;
            }
        }
    }

    return {
        editedPixels,
        editedShare: state.classMask.data.length ? (editedPixels / state.classMask.data.length) * 100 : 0,
        touchedClasses: [...touchedClassIds].map((id) => legendById.get(id)?.name || `Class ${id}`),
        lowConfidenceEditedPixels,
        lowConfidenceEditedShare: editedPixels ? (lowConfidenceEditedPixels / editedPixels) * 100 : 0,
    };
}

function composeMaskRgba() {
    const rgba = new Uint8ClampedArray(state.classMask.width * state.classMask.height * 4);
    for (let index = 0; index < state.classMask.data.length; index += 1) {
        const classId = Number(state.classMask.data[index]);
        const color = legendById.get(classId)?.color || [255, 255, 255];
        const target = index * 4;
        rgba[target] = color[0];
        rgba[target + 1] = color[1];
        rgba[target + 2] = color[2];
        rgba[target + 3] = 255;
    }
    return rgba;
}

function composeOverlayRgba() {
    const rgba = new Uint8ClampedArray(state.original.imageData.data.length);
    const source = state.original.imageData.data;
    const alpha = 0.55;
    for (let index = 0; index < state.classMask.data.length; index += 1) {
        const classId = Number(state.classMask.data[index]);
        const color = legendById.get(classId)?.color || [255, 255, 255];
        const target = index * 4;
        rgba[target] = Math.round(source[target] * (1 - alpha) + color[0] * alpha);
        rgba[target + 1] = Math.round(source[target + 1] * (1 - alpha) + color[1] * alpha);
        rgba[target + 2] = Math.round(source[target + 2] * (1 - alpha) + color[2] * alpha);
        rgba[target + 3] = 255;
    }
    return rgba;
}

function composeConfidenceRgba() {
    const rgba = new Uint8ClampedArray(state.original.imageData.data.length);
    for (let index = 0; index < state.confidenceMap.data.length; index += 1) {
        const confidence = state.confidenceMap.data[index];
        const target = index * 4;
        rgba[target] = confidence;
        rgba[target + 1] = Math.round(confidence * 0.45);
        rgba[target + 2] = Math.round(255 - confidence * 0.25);
        rgba[target + 3] = 255;
    }
    return rgba;
}

function composeUncertaintyRgba() {
    const rgba = new Uint8ClampedArray(state.original.imageData.data.length);
    const source = state.original.imageData.data;
    const threshold = currentConfidenceThreshold();
    for (let index = 0; index < state.confidenceMap.data.length; index += 1) {
        const confidence = state.confidenceMap.data[index] / 255;
        const target = index * 4;
        if (confidence < threshold) {
            rgba[target] = Math.round(source[target] * 0.42 + 214 * 0.58);
            rgba[target + 1] = Math.round(source[target + 1] * 0.42 + 131 * 0.58);
            rgba[target + 2] = Math.round(source[target + 2] * 0.42 + 56 * 0.58);
        } else {
            rgba[target] = source[target];
            rgba[target + 1] = source[target + 1];
            rgba[target + 2] = source[target + 2];
        }
        rgba[target + 3] = 255;
    }
    return rgba;
}

function renderCanvas() {
    showCanvasState();
    if (!state.original || !state.classMask) {
        const ctx = elements.canvas.getContext("2d");
        ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
        return;
    }

    const width = state.original.width;
    const height = state.original.height;
    elements.canvas.width = width;
    elements.canvas.height = height;
    elements.canvas.style.width = `${Math.round(width * currentZoom())}px`;
    elements.canvas.style.height = `${Math.round(height * currentZoom())}px`;

    let rgba;
    if (state.activeView === "mask") {
        rgba = composeMaskRgba();
    } else if (state.activeView === "confidence") {
        rgba = composeConfidenceRgba();
    } else if (state.activeView === "uncertainty") {
        rgba = composeUncertaintyRgba();
    } else if (state.activeView === "original") {
        rgba = new Uint8ClampedArray(state.original.imageData.data);
    } else {
        rgba = composeOverlayRgba();
    }

    const ctx = elements.canvas.getContext("2d");
    const imageData = new ImageData(rgba, width, height);
    ctx.putImageData(imageData, 0, 0);
}

function renderMetrics() {
    if (!state.prediction) {
        elements.metricsRibbon.innerHTML = "";
        return;
    }

    const summary = computeQaSummary();
    const confidence = state.prediction.confidence_summary || {};
    const metrics = [
        { label: "Edited pixels", value: summary.editedPixels.toLocaleString() },
        { label: "Edited share", value: formatPercent(summary.editedShare) },
        { label: "Mean confidence", value: Number(confidence.mean_confidence || 0).toFixed(3) },
        { label: "Low-conf edits", value: `${summary.lowConfidenceEditedPixels.toLocaleString()} | ${formatPercent(summary.lowConfidenceEditedShare)}` },
        { label: "Reviewer", value: elements.reviewerName.value || "Not set" },
        { label: "Status", value: elements.statusSelect.value },
    ];

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

function renderQaSummary() {
    if (!state.prediction) {
        elements.qaSummary.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__value">QA summary appears after a review run.</div>
            </article>
        `;
        return;
    }

    const summary = computeQaSummary();
    const cards = [
        { label: "Edited pixels", value: summary.editedPixels.toLocaleString() },
        { label: "Edited share", value: formatPercent(summary.editedShare) },
        { label: "Status", value: elements.statusSelect.value },
        { label: "Reviewer", value: elements.reviewerName.value || "Not set" },
    ];

    elements.qaSummary.innerHTML = cards
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

function renderTouchedClasses() {
    const summary = computeQaSummary();
    if (!summary.touchedClasses.length) {
        elements.touchedClasses.innerHTML = `<div class="pill">No edits yet</div>`;
        return;
    }

    elements.touchedClasses.innerHTML = summary.touchedClasses
        .map((className) => `<div class="pill is-good">${className}</div>`)
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
                        <div class="legend-item__meta">${row ? formatPercent(row.percentage) : "0.00%"}</div>
                    </div>
                </article>
            `;
        })
        .join("");
}

function renderStats(rows) {
    if (!rows.length) {
        elements.statsGrid.innerHTML = `
            <div class="empty-board">Corrected per-class statistics will appear here after inference.</div>
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

function correctedMaskDataUrl() {
    if (!state.classMask) {
        return "";
    }

    const canvas = document.createElement("canvas");
    canvas.width = state.classMask.width;
    canvas.height = state.classMask.height;
    const ctx = canvas.getContext("2d");
    const rgba = composeMaskRgba();
    ctx.putImageData(new ImageData(rgba, state.classMask.width, state.classMask.height), 0, 0);
    return canvas.toDataURL("image/png");
}

function reviewReportPayload(rows) {
    const summary = computeQaSummary();
    return {
        report_title: "SegImBud Review QA Report",
        mode_label: "Review and QA",
        summary_metrics: [
            { label: "Edited pixels", value: summary.editedPixels.toLocaleString() },
            { label: "Edited share", value: formatPercent(summary.editedShare) },
            { label: "Reviewer", value: elements.reviewerName.value || "Not set" },
            { label: "Status", value: elements.statusSelect.value },
        ],
        image_panels: [
            {
                title: "Original scene",
                image_base64: state.original.canvas.toDataURL("image/png").split(",")[1],
            },
            {
                title: "Reviewed overlay",
                image_base64: elements.canvas.toDataURL("image/png").split(",")[1],
            },
            {
                title: "Corrected mask",
                image_base64: correctedMaskDataUrl().split(",")[1],
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
            `Review status: ${elements.statusSelect.value}.`,
            `Reviewer: ${elements.reviewerName.value || "Not set"}.`,
            elements.notes.value || "No review notes recorded.",
        ],
    };
}

function renderExports(rows) {
    const ready = Boolean(state.prediction && state.classMask);
    elements.downloadReport.disabled = !ready;
    elements.downloadMask.disabled = !ready;
    elements.downloadStats.disabled = !ready;
    elements.downloadQa.disabled = !ready;

    if (!ready) {
        return;
    }

    elements.downloadReport.onclick = async () => {
        try {
            setStageStatus("Building QA PDF report...");
            const blob = await fetchBlob("/reports/pdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(reviewReportPayload(rows)),
            });
            downloadBlob("segimbud-review-report.pdf", blob, "application/pdf");
            setStageStatus("QA report downloaded.");
        } catch (error) {
            setStageStatus(error.message || "Report export failed.");
        }
    };

    elements.downloadMask.onclick = () => {
        const [prefix, content] = correctedMaskDataUrl().split(",");
        const mimeType = prefix.includes("image/png") ? "image/png" : "application/octet-stream";
        downloadBlob(
            "segimbud-reviewed-mask.png",
            Uint8Array.from(atob(content), (char) => char.charCodeAt(0)),
            mimeType,
        );
    };

    elements.downloadStats.onclick = () =>
        downloadBlob(
            "segimbud-reviewed-stats.csv",
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

    elements.downloadQa.onclick = () => {
        const summary = computeQaSummary();
        const qaPackage = {
            status: elements.statusSelect.value,
            reviewer: elements.reviewerName.value || "",
            notes: elements.notes.value || "",
            edited_pixels: summary.editedPixels,
            edited_percentage: Number(summary.editedShare.toFixed(2)),
            touched_classes: summary.touchedClasses,
            low_confidence_edited_pixels: summary.lowConfidenceEditedPixels,
            low_confidence_edited_percentage: Number(summary.lowConfidenceEditedShare.toFixed(2)),
            created_at: new Date().toISOString(),
        };
        downloadBlob("segimbud-review-qa.json", JSON.stringify(qaPackage, null, 2), "application/json");
    };
}

function renderAll() {
    const rows = computeReviewRows();
    renderCanvas();
    renderMetrics();
    renderQaSummary();
    renderTouchedClasses();
    renderLegend(rows);
    renderStats(rows);
    renderExports(rows);
}

function setView(view) {
    state.activeView = view;
    elements.viewButtons.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.view === view);
    });
    renderCanvas();
}

async function handleRun(event) {
    event?.preventDefault?.();
    const file = currentFile();
    if (!file) {
        setStageStatus("Choose an image before starting review.");
        return;
    }

    clearReviewState();
    setStageStatus(`Running prediction for ${file.name}...`);
    elements.runButton.disabled = true;

    try {
        const sourceUrl = URL.createObjectURL(file);
        state.original = {
            src: sourceUrl,
            ...(await imageDataFromSource(sourceUrl)),
        };

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
        state.originalMask = await grayscaleArrayFromBase64(payload.class_mask_image);
        state.classMask = cloneMask(state.originalMask);
        state.confidenceMap = await grayscaleArrayFromBase64(payload.confidence_map_image);
        renderAll();
        setView("overlay");
        setStageStatus("Review canvas ready. Paint corrections directly onto the mask.");
    } catch (error) {
        clearReviewState();
        renderAll();
        setStageStatus(error.message || "The review run failed.");
    } finally {
        elements.runButton.disabled = false;
    }
}

function resetWorkspace() {
    clearReviewState();
    state.selectedFile = null;
    state.activeView = "overlay";
    elements.form.reset();
    elements.resolution.value = "0.5";
    elements.statusSelect.value = "pending";
    elements.brushSize.value = "18";
    elements.zoom.value = "1";
    elements.confidenceThreshold.value = "0.6";
    elements.reviewerName.value = "";
    elements.notes.value = "";
    renderAll();
    setStageStatus("Review workspace reset.");
}

function handleFileSelection() {
    const file = elements.file.files?.[0] || null;
    clearReviewState();
    state.selectedFile = file;

    if (!file) {
        renderAll();
        setStageStatus("Waiting for review image.");
        return;
    }

    renderAll();
    setStageStatus(`Loaded ${file.name}. Ready to run review.`);
}

function onPointerDown(event) {
    if (!state.classMask) {
        return;
    }
    state.isPainting = true;
    elements.canvas.setPointerCapture(event.pointerId);
    paintCircle(event.clientX, event.clientY);
    renderCanvas();
}

function onPointerMove(event) {
    if (!state.isPainting) {
        return;
    }
    paintCircle(event.clientX, event.clientY);
    renderCanvas();
}

function onPointerUp(event) {
    if (!state.isPainting) {
        return;
    }
    state.isPainting = false;
    try {
        elements.canvas.releasePointerCapture(event.pointerId);
    } catch (error) {
        // no-op
    }
    renderAll();
}

function populateBrushClasses() {
    elements.brushClass.innerHTML = legend
        .map((entry) => `<option value="${entry.class_id}">${entry.name}</option>`)
        .join("");
    elements.brushClass.value = "1";
}

async function init() {
    state.health = await refreshGlobalHealth();
    populateBrushClasses();
    renderAll();
    elements.form.addEventListener("submit", handleRun);
    elements.file.addEventListener("change", handleFileSelection);
    elements.runButton.addEventListener("click", handleRun);
    elements.resetButton.addEventListener("click", resetWorkspace);
    elements.viewButtons.forEach((button) => {
        button.addEventListener("click", () => setView(button.dataset.view));
    });
    elements.zoom.addEventListener("input", renderCanvas);
    elements.confidenceThreshold.addEventListener("input", renderAll);
    elements.statusSelect.addEventListener("change", renderAll);
    elements.reviewerName.addEventListener("input", renderAll);
    elements.notes.addEventListener("input", renderAll);
    elements.canvas.addEventListener("pointerdown", onPointerDown);
    elements.canvas.addEventListener("pointermove", onPointerMove);
    elements.canvas.addEventListener("pointerup", onPointerUp);
    elements.canvas.addEventListener("pointerleave", onPointerUp);
}

init();
