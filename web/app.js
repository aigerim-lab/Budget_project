function getCategorySpending() {
  const inputs = Array.from(document.querySelectorAll(".cat"));
  const spend = {};
  inputs.forEach(inp => {
    spend[inp.dataset.cat] = Number(inp.value || 0);
  });
  return spend;
}

function renderAllocation(rows) {
  const tbody = document.querySelector("#allocTable tbody");
  tbody.innerHTML = "";
  rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.category}</td><td>${r.recommended_budget}</td>`;
    tbody.appendChild(tr);
  });
}

document.getElementById("btnPredict").addEventListener("click", async () => {
  const monthly_budget = Number(document.getElementById("budget").value || 0);
  const txn_count = Number(document.getElementById("txnCount").value || 1);
  const max_txn = Number(document.getElementById("maxTxn").value || 0);
  const category_spend = getCategorySpending();

  const payload = { monthly_budget, txn_count, max_txn, category_spend };

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await res.json();

  document.getElementById("riskProb").textContent = data.risk_prob.toFixed(3);
  document.getElementById("totalSpent").textContent = data.total_spent.toFixed(2);
  document.getElementById("budgetUsage").textContent = (data.budget_ratio * 100).toFixed(1) + "%";

  const badge = document.getElementById("statusBadge");
  badge.className = "badge " + (data.prediction === "RISK" ? "risk" : "ok");
  badge.textContent = data.prediction;

  renderAllocation(data.allocation);
});
