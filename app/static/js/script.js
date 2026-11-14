
"use strict";

/* =====================================================
   DEFECTAI - COMPLETE FRONTEND APPLICATION
   Dashboard + Prediction + History + SHAP + Analytics
===================================================== */


/* =====================================================
   1. API CONFIGURATION
===================================================== */

const API = {
    health: "/api/health",
    modelInfo: "/api/model-info",
    predict: "/api/predict",
    history: "/api/history",
    explain: "/api/explain",
    analytics: "/api/analytics"
};


/* =====================================================
   2. KC1 FEATURE DEFINITIONS
===================================================== */

const METRICS = [
    {
        name: "loc",
        label: "Lines of Code",
        description: "Total lines of code in the module."
    },
    {
        name: "v(g)",
        label: "Cyclomatic Complexity",
        description: "Cyclomatic complexity of the module."
    },
    {
        name: "ev(g)",
        label: "Essential Complexity",
        description: "Essential complexity of the module."
    },
    {
        name: "iv(g)",
        label: "Design Complexity",
        description: "Design complexity of the module."
    },
    {
        name: "n",
        label: "Total Operators and Operands",
        description: "Halstead program length."
    },
    {
        name: "v",
        label: "Volume",
        description: "Halstead software volume."
    },
    {
        name: "l",
        label: "Program Length Ratio",
        description: "Halstead program length ratio."
    },
    {
        name: "d",
        label: "Difficulty",
        description: "Halstead difficulty metric."
    },
    {
        name: "i",
        label: "Intelligence",
        description: "Halstead intelligence metric."
    },
    {
        name: "e",
        label: "Effort",
        description: "Halstead effort metric."
    },
    {
        name: "b",
        label: "Estimated Bugs",
        description: "Halstead estimated delivered bugs."
    },
    {
        name: "t",
        label: "Programming Time",
        description: "Halstead estimated programming time."
    },
    {
        name: "lOCode",
        label: "Lines of Code (Logical)",
        description: "Logical lines of source code."
    },
    {
        name: "lOComment",
        label: "Comment Lines",
        description: "Logical comment lines."
    },
    {
        name: "lOBlank",
        label: "Blank Lines",
        description: "Logical blank lines."
    },
    {
        name: "locCodeAndComment",
        label: "Code and Comment Lines",
        description: "Lines containing code and comments."
    },
    {
        name: "uniq_Op",
        label: "Unique Operators",
        description: "Number of unique operators."
    },
    {
        name: "uniq_Opnd",
        label: "Unique Operands",
        description: "Number of unique operands."
    },
    {
        name: "total_Op",
        label: "Total Operators",
        description: "Total operators in the module."
    },
    {
        name: "total_Opnd",
        label: "Total Operands",
        description: "Total operands in the module."
    },
    {
        name: "branchCount",
        label: "Branch Count",
        description: "Number of branches in the module."
    }
];


/* =====================================================
   3. GLOBAL SHAP FEATURE IMPORTANCE
===================================================== */

const SHAP_FEATURES = [
    { name: "lOCode", value: 0.072413 },
    { name: "total_Opnd", value: 0.037718 },
    { name: "v", value: 0.033380 },
    { name: "loc", value: 0.032309 },
    { name: "i", value: 0.031636 },
    { name: "total_Op", value: 0.020282 },
    { name: "n", value: 0.020201 },
    { name: "t", value: 0.019100 },
    { name: "e", value: 0.019058 },
    { name: "b", value: 0.018428 }
];


/* =====================================================
   4. DOM ELEMENTS
===================================================== */

const predictionForm = document.getElementById(
    "prediction-form"
);

const predictionResult = document.getElementById(
    "prediction-result"
);

const predictButton = document.getElementById(
    "predict-button"
);


/* =====================================================
   5. HELPER FUNCTIONS
===================================================== */

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[character]);
}


function showToast(message, type = "success") {
    const container = document.getElementById(
        "toast-container"
    );

    const toast = document.createElement("div");

    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3500);
}


async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            data.error || `Request failed (${response.status})`
        );
    }

    return data;
}


function formatDate(dateString) {
    const date = new Date(dateString);

    if (Number.isNaN(date.getTime())) {
        return "Unknown date";
    }

    return date.toLocaleString();
}


function percentage(value) {
    return `${(Number(value) * 100).toFixed(2)}%`;
}


function riskClass(risk) {
    return `risk-${String(risk).toLowerCase()}`;
}


function predictionClass(prediction) {
    return String(prediction).toLowerCase() === "defective"
        ? "badge-defective"
        : "badge-clean";
}


/* =====================================================
   6. NAVIGATION
===================================================== */

function navigateTo(pageName) {

    const validPages = [
        "dashboard",
        "predict",
        "history",
        "insights",
        "performance"
    ];

    if (!validPages.includes(pageName)) {
        return;
    }

    document.querySelectorAll(".page").forEach(page => {
        page.classList.remove("active");
    });

    document.querySelectorAll(".nav-item").forEach(item => {
        item.classList.remove("active");
    });

    const selectedPage = document.getElementById(
        `page-${pageName}`
    );

    if (!selectedPage) {
        showToast(
            `Page ${pageName} was not found.`,
            "error"
        );
        return;
    }

    selectedPage.classList.add("active");

    const selectedNav = document.querySelector(
        `.nav-item[data-page="${pageName}"]`
    );

    if (selectedNav) {
        selectedNav.classList.add("active");
    }

    const titles = {
        dashboard: "Dashboard",
        predict: "Defect Prediction",
        history: "Prediction History",
        insights: "Model Insights",
        performance: "Model Performance"
    };

    document.getElementById(
        "page-title"
    ).textContent = titles[pageName];

    document.getElementById(
        "breadcrumb-page"
    ).textContent = pageName.toUpperCase();

    document.getElementById(
        "sidebar"
    ).classList.remove("open");

    if (pageName === "history") {
        loadHistory();
    }

    if (pageName === "dashboard") {
        loadDashboard();
    }

    if (pageName === "performance") {
        loadAnalytics();
    }

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


document.querySelectorAll(".nav-item").forEach(item => {
    item.addEventListener("click", () => {
        navigateTo(item.dataset.page);
    });
});


document.querySelectorAll("[data-navigate]").forEach(button => {
    button.addEventListener("click", () => {
        navigateTo(button.dataset.navigate);
    });
});


document.getElementById("menu-toggle").addEventListener(
    "click",
    () => {
        document.getElementById(
            "sidebar"
        ).classList.toggle("open");
    }
);


/* =====================================================
   7. GENERATE METRIC INPUTS
===================================================== */

function renderMetricInputs() {
    const container = document.getElementById(
        "metrics-grid"
    );

    container.innerHTML = METRICS.map((metric, index) => `
        <div class="metric-field">

            <label for="metric-${index}">
                ${escapeHTML(metric.label)}
            </label>

            <input
                type="number"
                id="metric-${index}"
                name="${escapeHTML(metric.name)}"
                placeholder="Enter ${escapeHTML(metric.name)}"
                min="0"
                step="any"
                required
            >

            <small>
                ${escapeHTML(metric.description)}
            </small>

        </div>
    `).join("");
}


/* =====================================================
   8. CLEAR FORM
===================================================== */

document.getElementById("clear-form").addEventListener(
    "click",
    () => {
        predictionForm.reset();

        predictionResult.classList.add("hidden");

        showToast("All metric fields cleared.");
    }
);


/* =====================================================
   9. SUBMIT DEFECT PREDICTION
===================================================== */

predictionForm.addEventListener("submit", async event => {

    event.preventDefault();

    if (!predictionForm.reportValidity()) {
        return;
    }

    const formData = new FormData(predictionForm);

    const metrics = {};

    for (const metric of METRICS) {

        const rawValue = formData.get(metric.name);

        const value = Number(rawValue);

        if (
            rawValue === null ||
            rawValue.trim() === "" ||
            !Number.isFinite(value) ||
            value < 0
        ) {
            showToast(
                `Please enter a valid non-negative value for ${metric.name}.`,
                "error"
            );

            return;
        }

        metrics[metric.name] = value;
    }

    predictButton.disabled = true;

    predictButton.innerHTML = "Analyzing...";

    try {

        const result = await apiRequest(API.predict, {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                metrics: metrics
            })
        });

        // Display normal prediction.
        displayPrediction(result);

        // Generate individual SHAP explanation.
        await loadIndividualExplanation(metrics);

        // Update dashboard and history.
        await loadHistory();

        await loadDashboard();

        showToast(
            "Prediction completed successfully!"
        );

    } catch (error) {

        showToast(
            error.message,
            "error"
        );

    } finally {

        predictButton.disabled = false;

        predictButton.innerHTML =
            '<span>Analyze Software</span><span>→</span>';
    }
});


/* =====================================================
   10. DISPLAY PREDICTION RESULT
===================================================== */

function displayPrediction(result) {

    const probability = Number(
        result.defective_probability
    );

    const percent = probability * 100;

    const isDefective = (
        result.prediction === "Defective"
    );

    predictionResult.classList.remove("hidden");

    document.getElementById(
        "result-prediction"
    ).textContent = result.prediction;

    document.getElementById(
        "result-description"
    ).textContent = isDefective
        ? "The model classified this module as defective at the selected decision threshold."
        : "The model classified this module as non-defective at the selected decision threshold.";

    document.getElementById(
        "result-percentage"
    ).textContent = `${percent.toFixed(2)}%`;

    document.getElementById(
        "probability-fill"
    ).style.width = `${Math.max(0, Math.min(100, percent))}%`;

    document.getElementById(
        "result-risk"
    ).textContent = result.risk_category;

    document.getElementById(
        "result-threshold"
    ).textContent = Number(
        result.threshold
    ).toFixed(2);

    document.getElementById(
        "result-id"
    ).textContent = `#${result.prediction_id}`;

    const badge = document.getElementById(
        "result-badge"
    );

    badge.textContent = result.prediction.toUpperCase();

    badge.className =
        `result-badge ${predictionClass(result.prediction)}`;

    document.getElementById(
        "result-risk"
    ).className = riskClass(result.risk_category);

    predictionResult.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* =====================================================
   11. MODEL INFORMATION
===================================================== */

async function loadModelInfo() {

    try {

        const data = await apiRequest(API.modelInfo);

        document.getElementById(
            "model-feature-count"
        ).textContent = data.feature_count;

        document.getElementById(
            "model-threshold"
        ).textContent = Number(
            data.classification_threshold
        ).toFixed(2);

        if (data.feature_count !== METRICS.length) {

            showToast(
                "Warning: Frontend and model feature counts differ.",
                "error"
            );
        }

    } catch (error) {

        showToast(
            "Unable to load model information: " + error.message,
            "error"
        );
    }
}


/* =====================================================
   12. SERVER HEALTH
===================================================== */

async function checkHealth() {

    const statusText = document.getElementById(
        "connection-text"
    );

    const sidebarStatus = document.getElementById(
        "sidebar-status"
    );

    try {

        await apiRequest(API.health);

        statusText.textContent = "System Online";

        sidebarStatus.textContent = "Connected";

    } catch (error) {

        statusText.textContent = "Connection Failed";

        sidebarStatus.textContent = "Disconnected";

        showToast(
            "Cannot connect to the Flask backend.",
            "error"
        );
    }
}


/* =====================================================
   13. PREDICTION HISTORY
===================================================== */

async function loadHistory() {

    const tbody = document.getElementById(
        "history-table-body"
    );

    try {

        const data = await apiRequest(
            `${API.history}?limit=100`
        );

        const history = data.history || [];

        if (history.length === 0) {

            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="empty-table">
                        No prediction history available.
                    </td>
                </tr>
            `;

            return;
        }

        tbody.innerHTML = history.map(item => `
            <tr>

                <td>#${Number(item.id)}</td>

                <td>
                    ${escapeHTML(formatDate(item.created_at))}
                </td>

                <td>
                    <span class="prediction-badge ${predictionClass(item.prediction)}">
                        ${escapeHTML(item.prediction)}
                    </span>
                </td>

                <td>
                    ${percentage(item.defective_probability)}
                </td>

                <td>
                    <span class="risk-badge ${riskClass(item.risk_category)}">
                        ${escapeHTML(item.risk_category)}
                    </span>
                </td>

            </tr>
        `).join("");

    } catch (error) {

        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="empty-table">
                    Unable to load prediction history.
                </td>
            </tr>
        `;

        showToast(error.message, "error");
    }
}


document.getElementById("refresh-history").addEventListener(
    "click",
    async () => {

        await loadHistory();

        showToast("Prediction history refreshed.");
    }
);


/* =====================================================
   14. DASHBOARD STATISTICS
===================================================== */

async function loadDashboard() {

    try {

        const data = await apiRequest(
            `${API.history}?limit=200`
        );

        const history = data.history || [];

        const total = history.length;

        const defective = history.filter(
            item => item.prediction === "Defective"
        ).length;

        const clean = total - defective;

        const averageProbability = total > 0
            ? history.reduce(
                (sum, item) =>
                    sum + Number(item.defective_probability),
                0
            ) / total
            : 0;

        document.getElementById(
            "total-predictions"
        ).textContent = total;

        document.getElementById(
            "defective-count"
        ).textContent = defective;

        document.getElementById(
            "clean-count"
        ).textContent = clean;

        document.getElementById(
            "average-probability"
        ).textContent = percentage(averageProbability);

        renderRecentPredictions(
            history.slice(0, 5)
        );

    } catch (error) {

        showToast(
            "Unable to load dashboard statistics.",
            "error"
        );
    }
}


/* =====================================================
   15. RECENT PREDICTIONS
===================================================== */

function renderRecentPredictions(history) {

    const container = document.getElementById(
        "recent-predictions"
    );

    if (history.length === 0) {

        container.innerHTML = `
            <div class="empty-state">
                No predictions available yet.
                Start your first analysis.
            </div>
        `;

        return;
    }

    container.innerHTML = history.map(item => `
        <div class="recent-item">

            <div>

                <strong>
                    Prediction #${Number(item.id)}
                </strong>

                <p>
                    ${escapeHTML(formatDate(item.created_at))}
                </p>

            </div>

            <div class="recent-right">

                <strong>
                    ${percentage(item.defective_probability)}
                </strong>

                <p>
                    <span class="prediction-badge ${predictionClass(item.prediction)}">
                        ${escapeHTML(item.prediction)}
                    </span>
                </p>

            </div>

        </div>
    `).join("");
}


/* =====================================================
   16. GLOBAL SHAP FEATURE IMPORTANCE
===================================================== */

function renderFeatureImportance(targetId) {

    const container = document.getElementById(
        targetId
    );

    const maximum = Math.max(
        ...SHAP_FEATURES.map(feature => feature.value)
    );

    container.innerHTML = SHAP_FEATURES.map(feature => {

        const width = (
            feature.value / maximum
        ) * 100;

        return `
            <div class="feature-row">

                <span class="feature-name">
                    ${escapeHTML(feature.name)}
                </span>

                <div class="feature-track">

                    <div
                        class="feature-fill"
                        style="width: ${width}%"
                    ></div>

                </div>

                <span class="feature-value">
                    ${feature.value.toFixed(6)}
                </span>

            </div>
        `;

    }).join("");
}


/* =====================================================
   17. PHASE 5 - INDIVIDUAL SHAP EXPLANATION
===================================================== */

async function loadIndividualExplanation(metrics) {

    const container = document.getElementById(
        "individual-explanation"
    );

    const positiveContainer = document.getElementById(
        "shap-positive"
    );

    const negativeContainer = document.getElementById(
        "shap-negative"
    );

    container.classList.remove("hidden");

    positiveContainer.innerHTML = `
        <div class="shap-empty">
            Calculating individual SHAP contributions...
        </div>
    `;

    negativeContainer.innerHTML = `
        <div class="shap-empty">
            Waiting for explanation...
        </div>
    `;

    try {

        const data = await apiRequest(
            API.explain,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    metrics: metrics
                })
            }
        );

        document.getElementById(
            "shap-base"
        ).textContent = percentage(
            data.base_probability
        );

        document.getElementById(
            "shap-predicted"
        ).textContent = percentage(
            data.predicted_probability
        );

        renderShapContributions(
            data.positive_contributions,
            positiveContainer,
            "positive"
        );

        renderShapContributions(
            data.negative_contributions,
            negativeContainer,
            "negative"
        );

    } catch (error) {

        positiveContainer.innerHTML = `
            <div class="shap-empty">
                Individual explanation unavailable:
                ${escapeHTML(error.message)}
            </div>
        `;

        negativeContainer.innerHTML = "";

        showToast(
            "Prediction succeeded, but individual SHAP explanation failed.",
            "error"
        );
    }
}


function renderShapContributions(
    features,
    container,
    direction
) {

    if (!features || features.length === 0) {

        container.innerHTML = `
            <div class="shap-empty">
                No ${direction} contributions found.
            </div>
        `;

        return;
    }

    const className = direction === "positive"
        ? "shap-positive"
        : "shap-negative";

    container.innerHTML = features.map(feature => {

        const impact = Number(
            feature.impact_percentage_points
        );

        const sign = impact > 0 ? "+" : "";

        return `
            <div class="shap-item">

                <div>

                    <strong>
                        ${escapeHTML(feature.feature)}
                    </strong>

                    <small>
                        Input value:
                        ${Number(feature.input_value).toFixed(4)}
                    </small>

                </div>

                <div class="shap-impact ${className}">
                    ${sign}${impact.toFixed(3)} pp
                </div>

            </div>
        `;

    }).join("");
}


/* =====================================================
   18. PHASE 6 - MODEL ANALYTICS
===================================================== */

async function loadAnalytics() {

    const comparisonContainer = document.getElementById(
        "model-comparison-container"
    );

    const figureContainer = document.getElementById(
        "evaluation-figures"
    );

    comparisonContainer.innerHTML = `
        <div class="empty-state">
            Loading model evaluation results...
        </div>
    `;

    figureContainer.innerHTML = `
        <div class="empty-state">
            Loading evaluation figures...
        </div>
    `;

    try {

        const data = await apiRequest(API.analytics);

        renderModelComparisons(
            data.model_comparisons || []
        );

        renderEvaluationFigures(
            data.figures || []
        );

    } catch (error) {

        comparisonContainer.innerHTML = `
            <div class="analytics-empty">
                Could not load model comparison data:
                ${escapeHTML(error.message)}
            </div>
        `;

        figureContainer.innerHTML = `
            <div class="analytics-empty">
                Evaluation figures are unavailable.
            </div>
        `;

        showToast(
            "Could not load model analytics.",
            "error"
        );
    }
}


/* =====================================================
   19. RENDER MODEL COMPARISON TABLES
===================================================== */

function formatAnalyticsValue(value) {

    if (value === null || value === undefined) {
        return "--";
    }

    if (typeof value === "number") {

        return Number.isInteger(value)
            ? String(value)
            : value.toFixed(4);
    }

    return String(value);
}


function renderModelComparisons(comparisons) {

    const container = document.getElementById(
        "model-comparison-container"
    );

    if (!comparisons.length) {

        container.innerHTML = `
            <div class="analytics-empty">
                No model comparison CSV files were found in
                reports/generated.

                Check that tuned_model_comparison.csv exists.
            </div>
        `;

        return;
    }

    container.innerHTML = comparisons.map(comparison => {

        const rows = comparison.results || [];

        if (!rows.length) {
            return "";
        }

        const columns = Object.keys(rows[0]);

        const header = columns.map(column => `
            <th>
                ${escapeHTML(column)}
            </th>
        `).join("");

        const body = rows.map(row => `
            <tr>
                ${columns.map(column => `
                    <td>
                        ${escapeHTML(
                            formatAnalyticsValue(row[column])
                        )}
                    </td>
                `).join("")}
            </tr>
        `).join("");

        return `
            <h4 class="comparison-heading">
                ${escapeHTML(comparison.file)}
            </h4>

            <div class="table-container">

                <table class="comparison-table">

                    <thead>
                        <tr>${header}</tr>
                    </thead>

                    <tbody>
                        ${body}
                    </tbody>

                </table>

            </div>
        `;

    }).join("");

    if (!container.innerHTML.trim()) {

        container.innerHTML = `
            <div class="analytics-empty">
                No usable model comparison records were found.
            </div>
        `;
    }
}


/* =====================================================
   20. RENDER EVALUATION FIGURES
===================================================== */

function renderEvaluationFigures(figures) {

    const container = document.getElementById(
        "evaluation-figures"
    );

    if (!figures.length) {

        container.innerHTML = `
            <div class="analytics-empty">
                No evaluation images were found.

                Expected location:
                reports/generated/figures/
            </div>
        `;

        return;
    }

    const labels = {
        "confusion_matrix.png": "Confusion Matrix",
        "roc_curve.png": "ROC Curve",
        "precision_recall_curve.png": "Precision-Recall Curve"
    };

    container.innerHTML = figures.map(figure => `

        <div class="evaluation-figure">

            <h4>
                ${escapeHTML(
                    labels[figure.name] || figure.name
                )}
            </h4>

            <img
                src="${escapeHTML(figure.url)}"
                alt="${escapeHTML(figure.name)}"
                loading="lazy"
            >

        </div>

    `).join("");
}


/* =====================================================
   21. REFRESH ANALYTICS
===================================================== */

document.getElementById(
    "refresh-analytics"
).addEventListener(
    "click",
    async () => {

        await loadAnalytics();

        showToast(
            "Model analytics refreshed."
        );
    }
);


/* =====================================================
   22. INITIALIZE APPLICATION
===================================================== */

async function initializeApp() {

    renderMetricInputs();

    renderFeatureImportance(
        "dashboard-feature-list"
    );

    renderFeatureImportance(
        "insights-feature-list"
    );

    await Promise.all([
        checkHealth(),
        loadModelInfo(),
        loadDashboard(),
        loadHistory()
    ]);
}


document.addEventListener(
    "DOMContentLoaded",
    initializeApp
);
