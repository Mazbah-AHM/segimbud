import {
    buildCsv,
    classIdsFromSelection,
    createToggleMarkup,
    dataUrlFromBase64,
    downloadBase64,
    downloadBlob,
    fetchBlob,
    fetchJson,
    formatAreaKm2,
    formatPercent,
    grayscaleArrayFromBase64,
    imageDataFromSource,
    refreshGlobalHealth,
    renderOverlayDataUrl,
    setImageTarget,
} from "../core.js";

const legend = window.SEGIMBUD_CONFIG.legend || [];

const elements = {
    form: document.getElementById("change-form"),
    beforeFile: document.getElementById("change-before-file"),
    afterFile: document.getElementById("change-after-file"),
    resolution: document.getElementById("change-resolution"),
    overlayAlpha: document.getElementById("change-overlay-alpha"),
    classToggles: document.getElementById("change-class-toggles"),
    selectAll: document.getElementById("change-select-all"),
    resetButton: document.getElementById("change-reset-button"),
    runButton: document.getElementById("change-run-button"),
    runStatus: document.getElementById("change-run-status"),
    stageImage: document.getElementById("change-stage-image"),
    emptyState: document.getElementById("change-empty-state"),
    previewBefore: document.getElementById("change-preview-before"),
    previewAfter: document.getElementById("change-preview-after"),
    stageLabel: document.getElementById("change-stage-label"),
    viewButtons: [...document.querySelectorAll("#change-view-switcher [data-view]")],
    metricsRibbon: document.getElementById("change-metrics-ribbon"),
    areaSummary: document.getElementById("change-area-summary"),
    transitions: document.getElementById("change-transitions"),
    insights: document.getElementById("change-insights"),
    deltaGrid: document.getElementById("change-delta-grid"),
    downloadMask: document.getElementById("change-download-mask"),
    downloadTransitions: document.getElementById("change-download-transitions"),
    downloadDelta: document.getElementById("change-download-delta"),
    downloadReport: document.getElementById("change-download-report"),
};

const state = {
    health: null,
    beforeOriginal: null,
    afterOriginal: null,
    payload: null,
    beforeMask: null,
    afterMask: null,
    activeView: "change",
    visibleClasses: new Set(legend.map((entry) => entry.name)),
};

function clearChangeState() {
    state.payload = null;
    state.beforeMask = null;
    state.afterMask = null;
}

function currentResolution() {
    return Number.parseFloat(elements.resolution.value || "0") || 0;
}

function currentOverlayAlpha() {
    return Number.parseFloat(elements.overlayAlpha.value || "0.58") || 0.58;
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

function releaseOriginals() {
    if (state.beforeOriginal?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(state.beforeOriginal.src);
    }
    if (state.afterOriginal?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(state.afterOriginal.src);
    }
}

function releaseOriginal(kind) {
    const current = state[kind];
    if (current?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(current.src);
    }
}

function changeMetricPairs() {
    if (!state.payload) {
        return [];
    }

    const summary = state.payload.change_summary || {};
    const transitions = summary.transitions || [];
    const pixelArea = pixelAreaSqKm();
    const topTransition = transitions.length
        ? `${transitions[0].from_class} -> ${transitions[0].to_class}`
        : "No change";
    const metrics = [
        { label: "Changed pixels", value: Number(summary.changed_pixels || 0).toLocaleString() },
        { label: "Changed share", value: formatPercent(summary.changed_percentage || 0) },
        { label: "Top transition", value: topTransition },
        {
            label: "After confidence",
            value: Number(state.payload.after.confidence_summary?.mean_confidence || 0).toFixed(3),
        },
        {
            label: "Resolution",
            value: currentResolution() > 0 ? `${currentResolution().toFixed(2)} m/px` : "Set resolution",
        },
        {
            label: "Changed area",
            value:
                pixelArea !== null
                    ? `${(Number(summary.changed_pixels || 0) * pixelArea).toFixed(6)} km^2`
                    : "Awaiting scale",
        },
    ];
    return metrics;
}

function transitionRows() {
    const transitions = state.payload?.change_summary?.transitions || [];
    const pixelArea = pixelAreaSqKm();
    return transitions.map((transition) => ({
        ...transition,
        areaSqKm:
            transition.area_sq_km !== undefined
                ? Number(transition.area_sq_km)
                : pixelArea !== null
                  ? Number(transition.pixels || 0) * pixelArea
                  : null,
    }));
}

function buildStageViews() {
    if (!state.beforeOriginal || !state.afterOriginal) {
        return {};
    }

    const activeClassIds = visibleClassIds();
    const beforeOverlay =
        state.beforeMask && state.beforeOriginal
            ? renderOverlayDataUrl(state.beforeOriginal, state.beforeMask, legend, activeClassIds, currentOverlayAlpha())
            : state.beforeOriginal.src;
    const afterOverlay =
        state.afterMask && state.afterOriginal
            ? renderOverlayDataUrl(state.afterOriginal, state.afterMask, legend, activeClassIds, currentOverlayAlpha())
            : state.afterOriginal.src;

    return {
        change: state.payload?.change_mask_image ? dataUrlFromBase64(state.payload.change_mask_image) : "",
        "before-overlay": beforeOverlay,
        "after-overlay": afterOverlay,
        "before-confidence": state.payload?.before?.heatmap_image ? dataUrlFromBase64(state.payload.before.heatmap_image) : beforeOverlay,
        "after-confidence": state.payload?.after?.heatmap_image ? dataUrlFromBase64(state.payload.after.heatmap_image) : afterOverlay,
    };
}

function classDeltaRows() {
    const classDelta = state.payload?.change_summary?.class_delta || {};
    const pixelArea = pixelAreaSqKm();
    return Object.entries(classDelta)
        .map(([name, values]) => ({
            class: name,
            beforePixels: Number(values.before_pixels || 0),
            afterPixels: Number(values.after_pixels || 0),
            deltaPixels: Number(values.delta_pixels || 0),
            deltaPoints: Number(values.delta_percentage_points || 0),
            color: values.color || [255, 255, 255],
            deltaAreaSqKm:
                values.delta_area_sq_km !== undefined
                    ? Number(values.delta_area_sq_km)
                    : pixelArea !== null
                      ? Number(values.delta_pixels || 0) * pixelArea
                      : null,
        }))
        .sort((a, b) => Math.abs(b.deltaPixels) - Math.abs(a.deltaPixels));
}

function createChangeReportPayload() {
    const stageViews = buildStageViews();
    const rows = classDeltaRows();
    return {
        report_title: "SegImBud Change Detection Report",
        mode_label: "Change detection",
        summary_metrics: changeMetricPairs().slice(0, 4).map((metric) => ({
            label: metric.label,
            value: metric.value,
        })),
        image_panels: [
            {
                title: "Before scene",
                image_base64: dataToBase64(state.beforeOriginal?.canvas?.toDataURL("image/png") || ""),
            },
            {
                title: "After scene",
                image_base64: dataToBase64(state.afterOriginal?.canvas?.toDataURL("image/png") || ""),
            },
            {
                title: "Change mask",
                image_base64: dataToBase64(stageViews.change || ""),
            },
        ],
        insights: state.payload?.operational_insights || [],
        stats_rows: rows.map((row) => ({
            class: row.class,
            pixels: `${row.deltaPixels >= 0 ? "+" : ""}${row.deltaPixels.toLocaleString()}`,
            percentage: `${row.deltaPoints >= 0 ? "+" : ""}${row.deltaPoints.toFixed(2)} pp`,
            area_sq_km:
                row.deltaAreaSqKm !== null
                    ? `${row.deltaAreaSqKm >= 0 ? "+" : ""}${Number(row.deltaAreaSqKm).toFixed(6)}`
                    : "-",
        })),
        extra_notes: [
            `Active lens at export: ${state.activeView}.`,
            `Visible classes: ${[...state.visibleClasses].join(", ") || "None"}.`,
            currentResolution() > 0
                ? `Resolution used for area estimates: ${currentResolution().toFixed(2)} m/px.`
                : "Area estimates were not requested in this run.",
        ],
    };
}

function renderMetrics() {
    const metrics = changeMetricPairs();
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

function renderTransitions() {
    const transitions = transitionRows();
    if (!transitions.length) {
        elements.transitions.innerHTML = `
            <article class="transition-item">
                <div class="transition-item__title">No transitions detected yet.</div>
                <div class="transition-item__sub">Run change detection to populate the transition board.</div>
            </article>
        `;
        return;
    }

    elements.transitions.innerHTML = transitions
        .map(
            (transition) => `
                <article class="transition-item">
                    <div class="transition-item__head">
                        <div class="transition-item__title">${transition.from_class} -> ${transition.to_class}</div>
                        <div class="transition-item__value">${formatPercent(transition.percentage_of_image)}</div>
                    </div>
                    <div class="transition-item__sub">${Number(transition.pixels).toLocaleString()} pixels | ${formatAreaKm2(transition.areaSqKm)}</div>
                </article>
            `,
        )
        .join("");
}

function renderAreaSummary() {
    if (!state.payload) {
        elements.areaSummary.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__value">Run change detection to calculate area change.</div>
            </article>
        `;
        return;
    }

    if (currentResolution() <= 0) {
        elements.areaSummary.innerHTML = `
            <article class="runtime-card runtime-card--muted">
                <div class="runtime-card__label">Scale required</div>
                <div class="runtime-card__value">Enter meters per pixel to calculate area change.</div>
            </article>
        `;
        return;
    }

    const pixelArea = pixelAreaSqKm();
    const summary = state.payload.change_summary || {};
    const changedArea = Number(summary.changed_pixels || 0) * pixelArea;
    const sceneArea = state.beforeOriginal ? state.beforeOriginal.width * state.beforeOriginal.height * pixelArea : 0;
    const rows = classDeltaRows();
    const gainRow = [...rows].filter((row) => row.deltaAreaSqKm > 0).sort((a, b) => b.deltaAreaSqKm - a.deltaAreaSqKm)[0];
    const lossRow = [...rows].filter((row) => row.deltaAreaSqKm < 0).sort((a, b) => a.deltaAreaSqKm - b.deltaAreaSqKm)[0];

    const cards = [
        { label: "Resolution", value: `${currentResolution().toFixed(2)} m/px` },
        { label: "Scene area", value: `${sceneArea.toFixed(6)} km^2` },
        { label: "Changed area", value: `${changedArea.toFixed(6)} km^2` },
        { label: "Largest gain", value: gainRow ? `${gainRow.class} | +${Math.abs(gainRow.deltaAreaSqKm).toFixed(6)} km^2` : "-" },
        { label: "Largest loss", value: lossRow ? `${lossRow.class} | -${Math.abs(lossRow.deltaAreaSqKm).toFixed(6)} km^2` : "-" },
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
    const insights = state.payload?.operational_insights || [];
    elements.insights.innerHTML = insights.length
        ? insights
              .map(
                  (insight, index) => `
                    <article class="insight-card">
                        <div class="runtime-card__label">Insight ${index + 1}</div>
                        <div class="insight-card__body">${insight}</div>
                    </article>
                  `,
              )
              .join("")
        : `
            <article class="insight-card">
                <div class="insight-card__body">Change insights will appear here after a comparison run.</div>
            </article>
        `;
}

function renderDeltaGrid() {
    const rows = classDeltaRows();
    if (!rows.length) {
        elements.deltaGrid.innerHTML = `
            <div class="empty-board">Run change detection to populate per-class gains and losses.</div>
        `;
        return;
    }
    elements.deltaGrid.innerHTML = rows
        .map(
            (row) => `
                <article class="stat-card">
                    <div class="stat-card__head">
                        <div class="stat-card__title-wrap">
                            <span class="stat-card__swatch" style="background: rgb(${row.color[0]}, ${row.color[1]}, ${row.color[2]});"></span>
                            <span class="stat-card__title">${row.class}</span>
                        </div>
                        <span class="stat-card__value">${row.deltaPoints >= 0 ? "+" : ""}${row.deltaPoints.toFixed(2)} pp</span>
                    </div>
                    <div class="stat-card__meta">Pixel delta: ${row.deltaPixels >= 0 ? "+" : ""}${row.deltaPixels.toLocaleString()}</div>
                    <div class="stat-card__meta">Area delta: ${
                row.deltaAreaSqKm !== null
                    ? `${row.deltaAreaSqKm >= 0 ? "+" : ""}${Number(row.deltaAreaSqKm).toFixed(6)} km^2`
                    : "-"
            }</div>
                </article>
            `,
        )
        .join("");
}

function setStageStatus(message) {
    elements.runStatus.textContent = message;
}

function renderStage() {
    setImageTarget(elements.previewBefore, state.beforeOriginal?.src || "");
    setImageTarget(elements.previewAfter, state.afterOriginal?.src || "");

    if (!state.beforeOriginal && !state.afterOriginal) {
        setImageTarget(elements.stageImage, "", elements.emptyState);
        elements.stageLabel.textContent = "No comparison loaded";
        return;
    }

    const views = buildStageViews();
    const labels = {
        change: state.payload ? "Change mask" : "Awaiting change mask",
        "before-overlay": "Before overlay",
        "after-overlay": "After overlay",
        "before-confidence": "Before confidence",
        "after-confidence": "After confidence",
    };

    const stageSource = views[state.activeView] || state.afterOriginal?.src || state.beforeOriginal?.src || "";
    setImageTarget(elements.stageImage, stageSource, elements.emptyState);
    elements.stageLabel.textContent = labels[state.activeView];
}

function renderExports() {
    const ready = Boolean(state.payload);
    elements.downloadMask.disabled = !ready;
    elements.downloadTransitions.disabled = !ready;
    elements.downloadDelta.disabled = !ready;
    elements.downloadReport.disabled = !ready;

    if (!ready) {
        return;
    }

    elements.downloadMask.onclick = () => downloadBase64("segimbud-change-mask.png", state.payload.change_mask_image);
    elements.downloadTransitions.onclick = () =>
        downloadBlob(
            "segimbud-transitions.csv",
            buildCsv(transitionRows().map((row) => ({
                from_class: row.from_class,
                to_class: row.to_class,
                pixels: row.pixels,
                percentage_of_image: row.percentage_of_image,
                area_sq_km: row.areaSqKm !== null ? row.areaSqKm.toFixed(6) : "",
            })), [
                "from_class",
                "to_class",
                "pixels",
                "percentage_of_image",
                "area_sq_km",
            ]),
            "text/csv",
        );

    const deltaRows = classDeltaRows().map((row) => ({
        class: row.class,
        before_pixels: row.beforePixels,
        after_pixels: row.afterPixels,
        delta_pixels: row.deltaPixels,
        delta_percentage_points: row.deltaPoints,
        delta_area_sq_km: row.deltaAreaSqKm !== null ? row.deltaAreaSqKm.toFixed(6) : "",
    }));

    elements.downloadDelta.onclick = () =>
        downloadBlob(
            "segimbud-class-delta.csv",
            buildCsv(deltaRows, [
                "class",
                "before_pixels",
                "after_pixels",
                "delta_pixels",
                "delta_percentage_points",
                "delta_area_sq_km",
            ]),
            "text/csv",
        );

    elements.downloadReport.onclick = async () => {
        try {
            setStageStatus("Building PDF report...");
            const blob = await fetchBlob("/reports/pdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(createChangeReportPayload()),
            });
            downloadBlob("segimbud-change-report.pdf", blob, "application/pdf");
            setStageStatus("Change report downloaded.");
        } catch (error) {
            setStageStatus(error.message || "Report export failed.");
        }
    };
}

function renderAll() {
    renderMetrics();
    renderAreaSummary();
    renderTransitions();
    renderInsights();
    renderDeltaGrid();
    renderStage();
    renderExports();
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
    elements.selectAll?.addEventListener("click", () => {
        [...elements.classToggles.querySelectorAll("input[type='checkbox']")].forEach((input) => {
            input.checked = true;
        });
        syncVisibleClassesFromInputs();
    });
}

async function loadOriginal(file) {
    const url = URL.createObjectURL(file);
    return {
        src: url,
        ...(await imageDataFromSource(url)),
    };
}

async function syncOriginalSelections() {
    const beforeFile = elements.beforeFile.files?.[0];
    const afterFile = elements.afterFile.files?.[0];

    releaseOriginal("beforeOriginal");
    releaseOriginal("afterOriginal");
    state.beforeOriginal = beforeFile ? await loadOriginal(beforeFile) : null;
    state.afterOriginal = afterFile ? await loadOriginal(afterFile) : null;
}

async function handleSelectionChange() {
    clearChangeState();

    try {
        await syncOriginalSelections();
        renderAll();
        if (state.beforeOriginal || state.afterOriginal) {
            const loadedCount = Number(Boolean(state.beforeOriginal)) + Number(Boolean(state.afterOriginal));
            setStageStatus(
                loadedCount === 2
                    ? "Both scenes loaded. Ready to run change detection."
                    : "One scene loaded. Select the second image to continue.",
            );
        } else {
            setStageStatus("Waiting for comparison pair.");
        }
    } catch (error) {
        releaseOriginals();
        state.beforeOriginal = null;
        state.afterOriginal = null;
        renderAll();
        setStageStatus(error.message || "The selected image could not be loaded.");
    }
}

async function handleRun(event) {
    event.preventDefault();
    const beforeFile = elements.beforeFile.files?.[0];
    const afterFile = elements.afterFile.files?.[0];
    if (!beforeFile || !afterFile) {
        setStageStatus("Choose both scenes before running change detection.");
        return;
    }

    clearChangeState();
    setStageStatus("Running change detection...");
    elements.runButton.disabled = true;

    try {
        await syncOriginalSelections();
        renderAll();
        setView("change");
        const formData = new FormData();
        formData.append("before_file", beforeFile);
        formData.append("after_file", afterFile);
        if (currentResolution() > 0) {
            formData.append("resolution_m_per_px", String(currentResolution()));
        }
        state.payload = await fetchJson("/change-detection", {
            method: "POST",
            body: formData,
        });
        if (state.payload.before?.class_mask_image) {
            state.beforeMask = await grayscaleArrayFromBase64(state.payload.before.class_mask_image);
        }
        if (state.payload.after?.class_mask_image) {
            state.afterMask = await grayscaleArrayFromBase64(state.payload.after.class_mask_image);
        }
        renderAll();
        setView("change");
        setStageStatus("Change detection complete.");
    } catch (error) {
        renderAll();
        setStageStatus(error.message || "The comparison request failed.");
    } finally {
        elements.runButton.disabled = false;
    }
}

function resetWorkspace() {
    releaseOriginals();
    state.beforeOriginal = null;
    state.afterOriginal = null;
    clearChangeState();
    state.activeView = "change";
    elements.form.reset();
    [...elements.classToggles.querySelectorAll("input[type='checkbox']")].forEach((input) => {
        input.checked = true;
    });
    state.visibleClasses = new Set(legend.map((entry) => entry.name));
    renderAll();
    setStageStatus("Change workspace reset.");
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
    elements.beforeFile.addEventListener("change", handleSelectionChange);
    elements.afterFile.addEventListener("change", handleSelectionChange);
    elements.resetButton.addEventListener("click", resetWorkspace);
    elements.overlayAlpha.addEventListener("input", renderStage);
    elements.resolution.addEventListener("input", renderAll);
    elements.viewButtons.forEach((button) => {
        button.addEventListener("click", () => setView(button.dataset.view));
    });
}

init();
