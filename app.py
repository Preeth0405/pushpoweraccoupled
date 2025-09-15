# --- Imports ---
import pandas as pd
import streamlit as st
import numpy as np
import numpy_financial as npf
import plotly.express as px
import json

st.set_page_config(layout="wide")

with st.sidebar:
    st.image("image.png",width = 150)
    st.header("🔒 Secure Login")

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Login logic
    if not st.session_state.authenticated:
        password = st.text_input("Enter password", type="password")
        if password == "pushpower123":
            st.session_state.authenticated = True
            st.rerun()
    else:
        st.success("➜] Logged in")
        if st.button("⏻ Logout"):
            st.session_state.authenticated = False
            st.rerun()

# --- Access control ---
if not st.session_state.authenticated:
    st.warning("Please enter the correct password to access the app.")
    st.stop()

st.set_page_config(layout="wide")
st.title("⚡ AC-Coupled Solar + Battery Simulator")

# --- Upload/Download Input Parameters ---
st.sidebar.title("💾 Save or Load Inputs")

uploaded_params = st.sidebar.file_uploader("📤 Upload Parameters (.json)", type="json")
if uploaded_params:
    uploaded_config = json.load(uploaded_params)
    for k, v in uploaded_config.items():
        st.session_state[k] = v
    st.sidebar.success("Inputs loaded from file!")

with st.sidebar.expander("📖 Glossary & How-to", expanded=False):
    st.markdown("""
    ## ⚡ Key Terms
    - **DC Size (kW)**: Nominal capacity of the PV array (DC side).
    - **Base DC Size (kW)**: DC size assumed in the uploaded PV file. Used for scaling.
    - **Inverter Capacity (kW)**: Maximum AC output the inverter can deliver.
    - **Inverter Efficiency (%)**: Conversion efficiency from DC to AC.
    - **Export Limit (kW)**: Maximum power that can be exported to the grid at any time.
    - **Import Limit (kW)**: Maximum grid power allowed to be imported at any time.
    - **Battery Capacity (kWh)**: Usable energy storage per battery unit.
    - **Depth of Discharge (DoD, %)**: Maximum usable fraction of the battery (e.g. 95% DoD leaves 5% SOC reserve).
    - **SOC (State of Charge, %)**: Real-time energy stored in the battery relative to usable capacity.
    - **C-rate**: Defines the speed of charging/discharging relative to battery size (0.5C = full charge in 2 hours).
    - **PCS (Power Conversion System)**: Converter handling battery ↔ AC bus flows.
    - **PCS Efficiencies (%)**: Charge and discharge efficiencies applied to AC/DC conversion.

    ## 🛠 How the Simulation Works
    1. **Data Inputs**  
       - Load profile (CSV): Time series of site demand (kWh).  
       - PV output (CSV): Time series of PV production (kWh).  
       - Both must align in time resolution.  
       
    2. **Scaling PV**  
       - PV data is scaled from the base DC size in the file to the user-defined DC size.  
       
    3. **AC Bus Dispatch Priorities**  
       - **Step 1:** PV to load (via inverter).  
       - **Step 2:** Battery discharge to load (if load > PV).  
       - **Step 3:** Import from grid (if still deficit).  
       - **Step 4:** If PV surplus: charge battery (respecting PCS and C-rate).  
       - **Step 5:** Remaining surplus → export (limited by export cap).  
       - **Step 6:** Any further PV beyond export cap = Excess/curtailment.  
       
    4. **Losses Accounting**  
       - **Inverter Losses:** DC → AC efficiency.  
       - **Battery Losses:** Charging + discharging inefficiencies.  
       - **PCS Losses:** Additional AC/DC conversion inefficiencies.  
       - **Clipped Energy:** DC energy > inverter limit.  

    ## 📊 Financial Model
    - **Capex (£):** Based on DC size (capex/kW) + battery units.  
    - **O&M Costs (%):** Annual cost as % of Capex.  
    - **Degradation:** Optional PV degradation per year.  
    - **Tariff Escalation:** Import/export tariff growth per year.  
    - **Outputs:**  
      - Annual bill savings  
      - Export income  
      - O&M costs  
      - Cumulative cash flow  
      - IRR, ROI, Payback, LCOE  

    ## 📈 Outputs to Check
    - **Monthly Summary Table:** Energy balances by month.  
    - **Annual Summary (Metrics):** Renewable fraction, export %, losses, battery utilization, cycles.  
    - **Charts:**  
      - Load profile (average/peak).  
      - Battery SOC and flows.  
      - Energy flows over time.  
      - Monthly energy balance.  

    ## 💡 Tips
    - Ensure time steps in **Load** and **PV files** match (hourly, half-hourly, etc.).  
    - Use realistic C-rates (0.5C → standard lithium-ion).  
    - Export/import limits mimic grid constraints.  
    - Excess energy highlights potential clipping or curtailment.  
    - For sensitivity studies, use **Batch Simulation** to test multiple system sizes.  
    """)


# --- Upload Section ---
st.header("1. Upload Input Data")
col1, col2 = st.columns(2)
with col1:
    load_file = st.file_uploader("Load Profile (CSV)", type="csv")
with col2:
    pv_file = st.file_uploader("PV Output (CSV)", type="csv")

# --- System Inputs ---
st.header("2. System Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    dc_size = st.number_input("DC System Size (kW)", value=st.session_state.get("dc_size", 40.0))
    base_dc_size = st.number_input("Base DC Size in PV File (kW)", value=st.session_state.get("base_dc_size", 40.0))
with col2:
    inverter_capacity = st.number_input("Inverter Capacity (kW)", value=st.session_state.get("inverter_capacity", 30.0))
    inverter_eff = st.number_input("Inverter Efficiency (%)", value=st.session_state.get("inverter_eff", 98.0)) / 100
with col3:
    export_limit = st.number_input("Export Limit (kW)", value=st.session_state.get("export_limit", 30.0))
    import_limit = st.number_input("Import Limit (kW)", value=st.session_state.get("import_limit", 100.0))

# --- Utility Rates ---
st.header("3. Utility Tariff Inputs")
col1, col2 = st.columns(2)
with col1:
    import_rate = st.number_input("Import rate (£/kWh)", min_value=0.1, value=st.session_state.get("import_rate", 0.25),
                                  step=0.01)
with col2:
    export_rate = st.number_input("Export rate (£/kWh)", min_value=0.00,
                                  value=st.session_state.get("export_rate", 0.05), step=0.005)

# --- Financial Parameters ---
st.header("4. Financial Assumptions")
col1, col2, col3 = st.columns(3)
with col1:
    capex_per_kw = st.number_input("Capex (Cost per kW)", value=st.session_state.get("capex_per_kw", 650.0))
    cost_of_battery = st.number_input("Battery Capex (Cost per Battery)",
                                      value=st.session_state.get("cost_of_battery", 20000.0))
    o_and_m_rate = st.number_input("O&M Cost (% of Capex per year)",
                                   value=st.session_state.get("o_and_m_rate", 1.0)) / 100
with col2:
    apply_degradation = st.checkbox("Apply Degradation", value=st.session_state.get("apply_degradation", False))
    degradation_rate = st.number_input("Degradation per Year (%)",
                                       value=st.session_state.get("degradation_rate", 0.4)) / 100
with col3:
    import_esc = st.number_input("Import Tariff Escalation (%/year)",
                                 value=st.session_state.get("import_esc", 2.0)) / 100
    export_esc = st.number_input("Export Tariff Escalation (%/year)",
                                 value=st.session_state.get("export_esc", 1.0)) / 100
    inflation = st.number_input("General Inflation Rate (%/year)", value=st.session_state.get("inflation", 3.0)) / 100

# --- Battery + PCS Inputs ---
st.header("5. Battery + PCS Configuration")
with st.expander("🔋 Battery + PCS Settings"):
    battery_qty = st.number_input("Battery Quantity", value=st.session_state.get("battery_qty", 1))
    battery_capacity = st.number_input("Battery Capacity per Unit (kWh)",
                                       value=st.session_state.get("battery_capacity", 50.0))
    dod = st.number_input("Depth of Discharge (%)", value=st.session_state.get("dod", 95.0)) / 100
    min_soc = st.number_input("Minimum SOC (%)", value=st.session_state.get("min_soc", 5.0)) / 100
    initial_soc = st.number_input("Initial SOC (%)", value=st.session_state.get("initial_soc", 100.0)) / 100
    c_rate = st.number_input("Battery C-rate", value=st.session_state.get("c_rate", 0.5))
    battery_eff = st.number_input("Battery Round-Trip Efficiency (%)",
                                  value=st.session_state.get("battery_eff", 96.0)) / 100

    # NEW: PCS Settings
    pcs_capacity = st.number_input("PCS Capacity (kW)", value=st.session_state.get("pcs_capacity", 30.0))
    pcs_eff_charge = st.number_input("PCS Charge Efficiency (%)",
                                     value=st.session_state.get("pcs_eff_charge", 98.0)) / 100
    pcs_eff_discharge = st.number_input("PCS Discharge Efficiency (%)",
                                        value=st.session_state.get("pcs_eff_discharge", 98.0)) / 100

# --- Save Current Input Parameters ---
if st.sidebar.button("📥 Save Inputs"):
    input_params = {
        "dc_size": dc_size,
        "base_dc_size": base_dc_size,
        "inverter_capacity": inverter_capacity,
        "inverter_eff": inverter_eff*100,
        "export_limit": export_limit,
        "import_limit": import_limit,
        "import_rate": import_rate,
        "export_rate": export_rate,
        "capex_per_kw": capex_per_kw,
        "cost_of_battery": cost_of_battery,
        "o_and_m_rate": o_and_m_rate*100,
        "apply_degradation": apply_degradation,
        "degradation_rate": degradation_rate*100,
        "import_esc": import_esc*100,
        "export_esc": export_esc*100,
        "inflation": inflation*100,
        "battery_qty": battery_qty,
        "battery_capacity": battery_capacity,
        "dod": dod*100,
        "min_soc": min_soc*100,
        "initial_soc": initial_soc*100,
        "c_rate": c_rate,
        "battery_eff": battery_eff*100,
        "pcs_capacity":pcs_capacity,
        "pcs_eff_charge":pcs_eff_charge*100,
        "pcs_eff_discharge":pcs_eff_discharge*100
        
        
    }

    json_string = json.dumps(input_params, indent=2)
    st.sidebar.download_button("⬇️ Download JSON", json_string, file_name="saved_inputs.json", mime="application/json")

# --- Simulation Execution ---
if load_file and pv_file:
    load_df = pd.read_csv(load_file)
    pv_df = pd.read_csv(pv_file)

    df = pd.DataFrame()
    df["Time"] = pd.to_datetime(load_df.iloc[:, 0], dayfirst=True)
    df["Load"] = load_df.iloc[:, 1]
    df["PV_base"] = pv_df.iloc[:, 1]
    df["Month"] = df["Time"].dt.to_period("M")
    df["Hour"] = df["Time"].dt.strftime("%H:%M")

    scaling = dc_size / base_dc_size
    df["PV Production"] = df["PV_base"] * scaling
    usable_capacity = battery_capacity * dod * battery_qty
    soc = usable_capacity * initial_soc

    avg_profile = df.groupby("Hour")["Load"].mean().reset_index(name="Average Load")
    peak_profile = df.groupby("Hour")["Load"].max().reset_index(name="Peak Load")

    total_discharge = 0
    charge_eff = np.sqrt(battery_eff)
    discharge_eff = np.sqrt(battery_eff)

    results = {
        "SOC Before Step (%)": [],
        "PV to Load": [],
        "PV to Load [AC]": [],
        "Battery Charge [Useful]": [],
        "Battery Charge [Raw AC Input]": [],
        "Battery Discharge [Raw]": [],
        "Battery Discharge [Useful]": [],
        "Battery Discharge to Load [AC]": [],
        "SOC (%)": [],
        "Import": [],
        "Export": [],
        "Excess": [],
        "Battery Losses": [],
        "PCS Losses": [],
        "PCS In (Charging)": [],
        "PCS Out (Discharging)": [],
        "Inverter Losses": [],
        "Clipped": [],
        "AC Bus Output": [],
        "AC Bus Balance Error": [],
        "Useful PV Production":[]
    }

    for i in df.index:
        pv = df.at[i, "PV Production"]
        load = df.at[i, "Load"]

        max_charge_pcs = pcs_capacity
        max_discharge_pcs = pcs_capacity

        max_charge_batt = battery_capacity * battery_qty * c_rate
        max_discharge_batt = battery_capacity * battery_qty * c_rate

        max_charge_possible = min(max_charge_pcs, max_charge_batt, usable_capacity - soc)
        max_discharge_possible = min(max_discharge_pcs, max_discharge_batt, soc - usable_capacity * min_soc)

        # --- Store SOC before step ---
        results["SOC Before Step (%)"].append((soc / usable_capacity) * 100)

        # --- PV → Inverter → AC Bus ---
        e_inv = min(pv, inverter_capacity)
        e_use_ac = e_inv * inverter_eff
        inv_losses = e_inv - e_use_ac

        # --- AC Bus dispatch priorities ---
        # Priority 1: PV AC to Load
        pv_to_load_ac = min(e_use_ac, load)
        remaining_load_ac = max(0, load - pv_to_load_ac)

        # --- Battery Discharge (raw → output) ---
        pcs_remaining_capacity = pcs_capacity

        # Step 1: Max raw discharge based on SOC and C-rate
        max_raw_discharge = min(max_discharge_possible, soc - usable_capacity * min_soc)
        max_raw_discharge = max(0, max_raw_discharge)

        # Step 2: Convert to possible AC output
        max_ac_output_from_battery = max_raw_discharge * pcs_eff_discharge * discharge_eff

        # Step 3: Actual needed AC output
        battery_discharge_to_load_ac = min(remaining_load_ac, pcs_remaining_capacity, max_ac_output_from_battery)
        pcs_remaining_capacity -= battery_discharge_to_load_ac

        # Step 4: Actual raw discharge used
        useful_discharge = battery_discharge_to_load_ac
        raw_discharge = useful_discharge / (pcs_eff_discharge * discharge_eff)
        soc -= raw_discharge
        soc = max(soc, usable_capacity * min_soc)
        total_discharge += useful_discharge

        # Track PCS Output (battery discharge)
        pcs_out = useful_discharge / discharge_eff  # raw discharge DC → PCS AC output before eff loss
        pcs_discharge_losses = pcs_out * (1 - pcs_eff_discharge)

        # --- Battery Charge (via PCS) ---
        surplus_ac = max(0, e_use_ac - pv_to_load_ac)
        raw_charge = min(surplus_ac, pcs_remaining_capacity, max_charge_possible)
        useful_charge = raw_charge * pcs_eff_charge * charge_eff
        soc += useful_charge

        pcs_in = raw_charge  # raw AC energy used to charge battery
        pcs_charge_losses = pcs_in * (1 - pcs_eff_charge)

        pcs_losses = pcs_discharge_losses + pcs_charge_losses
        battery_losses = (raw_charge - useful_charge) + (raw_discharge - useful_discharge)

        # --- Import ---
        import_energy = max(0, load - pv_to_load_ac - battery_discharge_to_load_ac)

        # --- Export ---
        pv_after_load_charge = max(0, e_use_ac - pv_to_load_ac - raw_charge)
        export = min(pv_after_load_charge, export_limit)

        # --- Excess ---
        excess = max(0, pv_after_load_charge - export)

        # --- Clipped ---
        clipped = max(0, pv - e_inv)

        # --- AC Bus Output calculation ---
        # Now including raw_charge and excess → correct
        ac_bus_output = pv_to_load_ac + battery_discharge_to_load_ac + export + raw_charge + excess
        ac_bus_expected_output = e_use_ac + battery_discharge_to_load_ac
        ac_bus_balance_error = ac_bus_expected_output - ac_bus_output

        # --- Store results ---
        pv_load_dc = pv_to_load_ac / inverter_eff
        useful_energy = pv - clipped - excess

        results["PV to Load"].append(pv_load_dc)
        results["PV to Load [AC]"].append(pv_to_load_ac)
        results["Battery Charge [Useful]"].append(useful_charge)
        results["Battery Charge [Raw AC Input]"].append(raw_charge)
        results["Battery Discharge [Raw]"].append(raw_discharge)
        results["Battery Discharge [Useful]"].append(useful_discharge)
        results["Battery Discharge to Load [AC]"].append(battery_discharge_to_load_ac)
        results["SOC (%)"].append((soc / usable_capacity) * 100)
        results["Import"].append(min(import_energy, import_limit))
        results["Export"].append(export)
        results["Excess"].append(excess)
        results["Battery Losses"].append(battery_losses)
        results["PCS Losses"].append(pcs_losses)
        results["PCS In (Charging)"].append(pcs_in)
        results["PCS Out (Discharging)"].append(pcs_out)
        results["Inverter Losses"].append(inv_losses)
        results["Clipped"].append(clipped)
        results["AC Bus Output"].append(ac_bus_output)
        results["AC Bus Balance Error"].append(ac_bus_balance_error)
        results["Useful PV Production"].append(useful_energy)

    # Attach results to dataframe
    for key in results:
        df[key] = results[key]

    # --- Monthly Summary ---
    monthly = df.groupby("Month").agg({
        "Load": "sum", "PV Production": "sum", "Useful PV Production":"sum", "PV to Load": "sum",
        "Battery Discharge [Useful]": "sum", "Import": "sum", "Export": "sum",
        "Excess": "sum", "Battery Losses": "sum", "Inverter Losses": "sum"
    }).rename(columns={
        "Load": "Load", "PV Production": "Production", "Useful PV Production": "Useful PV Energy", "PV to Load": "Solar On-site",
        "Battery Discharge [Useful]": "Battery", "Import": "Grid", "Export": "Export",
        "Excess": "Excess", "Battery Losses": "Battery Losses", "Inverter Losses": "Inverter Losses"
    }).reset_index()

    monthly["Month"] = monthly["Month"].astype(str)

    with st.expander("📅 Monthly Summary Table", expanded=False):
        st.dataframe(monthly)

    # --- Financial Projection ---
    st.header("6. 25-Year Financial Results")
    initial_capex = (dc_size * capex_per_kw) + (battery_qty * cost_of_battery)
    years = list(range(26))
    degradation_factors = [(1 - degradation_rate) ** (y - 1) if apply_degradation and y > 0 else 1.0 for y in years]
    cashflow = []
    cumulative = -initial_capex

    for y in years:
        if y == 0:
            cashflow.append({
                "Year": 0,
                "System Price (£)": -initial_capex,
                "O&M Costs (£)": 0,
                "Net Bill Savings (£)": 0,
                "Export Income (£)": 0,
                "Annual Cash Flow (£)": -initial_capex,
                "Cumulative Cash Flow (£)": -initial_capex
            })
            continue

        total_pv = df['PV Production'].sum()
        total_load = df['Load'].sum()

        deg = degradation_factors[y]
        pv_prod = total_pv * deg

        base_self_use_ratio = (df['PV to Load'].sum() + df['Battery Discharge [Useful]'].sum()) / df[
            'PV Production'].sum()
        pv_to_load = pv_prod * base_self_use_ratio

        base_export_ratio = df['Export'].sum() / pv_prod
        pv_export = pv_prod * base_export_ratio

        renewable_fraction = ((df['PV to Load'].sum() + df['Battery Discharge [Useful]'].sum()) / df[
            'Load'].sum()) * 100
        import_required = total_load - pv_to_load

        import_to_load = (df['Import'].sum() / df['Load'].sum()) * 100
        yearly_savings = ((df['PV to Load'].sum() + df['Battery Discharge [Useful]'].sum()) * import_rate) + (
                    df['Export'].sum() * export_rate) - (initial_capex * o_and_m_rate)

        imp_price = import_rate * ((1 + import_esc) ** (y - 1))
        exp_price = export_rate * ((1 + export_esc) ** (y - 1))

        savings = (total_load - import_required) * imp_price
        export_income = pv_export * exp_price
        om = initial_capex * o_and_m_rate * ((1 + inflation) ** (y - 1))

        annual_cashflow = savings + export_income - om
        cumulative += annual_cashflow

        cashflow.append({
            "Year": y,
            "System Price (£)": -initial_capex if y == 0 else 0,
            "O&M Costs (£)": -om if y > 0 else 0,
            "Net Bill Savings (£)": savings,
            "Export Income (£)": export_income,
            "Annual Cash Flow (£)": annual_cashflow,
            "Cumulative Cash Flow (£)": cumulative
        })

    fin_df = pd.DataFrame(cashflow)
    irr = npf.irr(fin_df['Annual Cash Flow (£)'])
    roi = (fin_df['Cumulative Cash Flow (£)'].iloc[-1] + initial_capex) / initial_capex

    payback = None
    payback_display = "Not achieved"
    for i in range(1, len(fin_df)):
        if fin_df.loc[i, 'Cumulative Cash Flow (£)'] >= 0:
            prev_cum = fin_df.loc[i - 1, 'Cumulative Cash Flow (£)']
            annual_cash = fin_df.loc[i, 'Annual Cash Flow (£)']
            if annual_cash != 0:
                payback = i - 1 + abs(prev_cum) / annual_cash
                years = int(payback)
                months = int(round((payback - years) * 12))
                payback_display = f"{years} years {months} months"
            break

    lcoe = initial_capex / sum([total_pv * d for d in degradation_factors[1:]])

    # --- Financial Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Capex (£)", f"{initial_capex:,.2f}")
    col2.metric("Payback Period", payback_display)
    col3.metric("IRR (%)", f"{irr * 100:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("LCOE (£/kWh)", f"{lcoe:.2f}")
    col5.metric("ROI (%)", f"{roi * 100:.2f}")
    col6.metric("First Year Savings (£)", f"{yearly_savings:.2f}")

    with st.expander("📋 Show Cash Flow Table"):
        st.dataframe(fin_df.style.format({
            "System Price (£)": "£{:,.2f}",
            "O&M Costs (£)": "£{:,.2f}",
            "Net Bill Savings (£)": "£{:,.2f}",
            "Export Income (£)": "£{:,.2f}",
            "Annual Cash Flow (£)": "£{:,.2f}",
            "Cumulative Cash Flow (£)": "£{:,.2f}"
        }))

    # --- Annual Summary (Metrics) ---
    with st.expander("📊 Annual Summary (Metrics)"):
        total = df[["Load", "Useful PV Production", "PV to Load [AC]", "Battery Discharge [Useful]", "Import", "Export", "Excess",
                    "Battery Losses", "Inverter Losses"]].sum()

        renewable_fraction = ((total['PV to Load [AC]'] + total['Battery Discharge [Useful]']) / total['Load']) * 100
        import_to_load = (total['Import'] / total['Load']) * 100
        export_to_grid = (total['Export'] / total[' Useful PV Production']) * 100
        battery_utilization = (total['Battery Discharge [Useful]'] / (usable_capacity * 365)) * 100
        battery_cycles = total_discharge / usable_capacity

        row1 = st.columns(4)
        row1[0].metric("🔌 Total Load (kWh)", f"{total['Load']:.2f}")
        row1[1].metric("🔄 Solar + Battery On-site (kWh)",
                       f"{total['PV to Load [AC]'] + total['Battery Discharge [Useful]']:.2f}")
        row1[2].metric("⚡ Grid Import (kWh)", f"{total['Import']:.2f}")
        row1[3].metric("📤 Exported (kWh)", f"{total['Export']:.2f}")

        row2 = st.columns(4)
        row2[0].metric("☀️ Useful PV Production (kWh)", f"{total['USeful PV Production']:.2f}")
        row2[1].metric("🔄 Solar + Battery On-site (%)", f"{renewable_fraction:.2f}%")
        row2[2].metric("⚡ Grid Import (%)", f"{import_to_load:.2f}%")
        row2[3].metric("📤 Exported (%)", f"{export_to_grid:.2f}%")

        row3 = st.columns(4)
        row3[0].metric("🌞 Solar On-site (kWh)", f"{total['PV to Load [AC]']:.2f}")
        row3[1].metric("🔋 Battery Use (kWh)", f"{total['Battery Discharge [Useful]']:.2f}")
        row3[2].metric("🗑️ Excess Energy (kWh)", f"{total['Excess']:.2f}")
        row3[3].metric("🔻 Inverter Losses (kWh)", f"{total['Inverter Losses']:.2f}")

        row4 = st.columns(4)
        row4[0].metric("🌞 Solar On-site (%)", f"{(total['PV to Load [AC]'] / total[' Useful PV Production']) * 100:.2f}%")
        row4[1].metric("🔋 Battery Use (%)",
                       f"{(total['Battery Discharge [Useful]'] / total[' USeful PV Production']) * 100:.2f}%")
        row4[2].metric("🗑️ Excess Energy (%)", f"{(total['Excess'] / total[' Useful PV Production']) * 100:.2f}%")
        row4[3].metric("🔻 Inverter Losses (%)", f"{(total['Inverter Losses'] / total['Useful PV Production']) * 100:.2f}%")

        row5 = st.columns(4)
        row5[0].metric("🔻 Battery Losses (kWh)", f"{total['Battery Losses']:.2f}")
        row5[1].metric("🔻 Battery Losses (%)", f"{(total['Battery Losses'] / total[' Useful PV Production']) * 100:.2f}%")
        row5[2].metric("🔁 Battery Cycles", f"{battery_cycles:.2f}")
        row5[3].metric("🔋📈 Battery Utilization (%)", f"{battery_utilization:.2f}")

    # --- Load Profile Charts ---
    with st.expander(" 📈 Load Profile"):
        st.plotly_chart(px.line(avg_profile, x="Hour", y="Average Load", title="Average Load Over Time"),
                        use_container_width=True)
        st.plotly_chart(px.line(peak_profile, x="Hour", y="Peak Load", title="Peak Load Over Time"),
                        use_container_width=True)

    # --- Battery SOC + Charge/Discharge Charts ---
    with st.expander("🔋 Battery Charts"):
        st.plotly_chart(px.line(df, x="Time", y="SOC (%)", title="Battery SOC Over Time"), use_container_width=True)
        st.plotly_chart(px.line(df, x="Time", y=["Battery Charge [Useful]", "Battery Discharge [Useful]"],
                                title="Battery Charge & Discharge"), use_container_width=True)

    # --- Renewable Energy Charts ---
    with st.expander("☀️ Renewable Energy Charts"):
        st.plotly_chart(px.line(df, x="Time",
                                y=["Load", "PV Production", "PV to Load", "Battery Discharge [Useful]","Battery Charge [Useful]","Import",
                                   "Export", "Excess"], title="Load vs System Flows"), use_container_width=True)
        st.plotly_chart(
            px.bar(monthly, x="Month", y=["Load", "Production", "Export", "Excess", "Grid"], barmode="group",
                   title="Monthly Solar Use & Export"), use_container_width=True)

    # --- Download CSV ---
    with st.expander("📥 Download Results"):
        st.download_button("Download CSV", df.to_csv(index=False), "final_ac_coupled_simulation.csv", "text/csv",
                           key="download_simulation_result")

    import datetime


    # =======================================
    # --- Batch Simulation (Separate Block) -
    # =======================================

    def simulate_one_system(dc_size, inverter_capacity, battery_capacity, pcs_capacity,
                            inverter_eff, battery_eff, pcs_eff,
                            base_dc_size, df, dod, min_soc,
                            initial_soc, battery_qty, c_rate,
                            export_limit, import_limit):
        """Run the full step-by-step simulation for one system config (batch version)."""

        scaling = dc_size / base_dc_size
        df_sim = df.copy()
        df_sim["PV Production"] = df_sim["PV_base"] * scaling

        usable_capacity = battery_capacity * dod
        soc = usable_capacity * initial_soc

        charge_eff = np.sqrt(battery_eff)
        discharge_eff = np.sqrt(battery_eff)

        total_discharge = 0
        pv_to_load_total = 0
        batt_use_total = 0
        import_total = 0
        export_total = 0
        excess_total = 0
        pv_prod_tot = 0

        for i in df_sim.index:
            pv = df_sim.at[i, "PV Production"]
            load = df_sim.at[i, "Load"]

            pv_prod_tot += pv

            # --- Limits ---
            max_charge_pcs = pcs_capacity
            max_discharge_pcs = pcs_capacity
            max_charge_batt = battery_capacity * battery_qty * c_rate
            max_discharge_batt = battery_capacity * battery_qty * c_rate
            max_charge_possible = min(max_charge_pcs, max_charge_batt, usable_capacity - soc)
            max_discharge_possible = min(max_discharge_pcs, max_discharge_batt, soc - usable_capacity * min_soc)

            # --- PV → Inverter ---
            e_inv = min(pv, inverter_capacity)
            e_use_ac = e_inv * inverter_eff

            # --- PV to Load ---
            pv_to_load_ac = min(e_use_ac, load)
            remaining_load_ac = max(0, load - pv_to_load_ac)

            # --- Battery Discharge ---
            max_raw_discharge = min(max_discharge_possible, soc - usable_capacity * min_soc)
            max_raw_discharge = max(0, max_raw_discharge)
            max_ac_output = max_raw_discharge * pcs_eff * discharge_eff
            batt_discharge_to_load = min(remaining_load_ac, pcs_capacity, max_ac_output)

            raw_discharge = batt_discharge_to_load / (pcs_eff * discharge_eff) if batt_discharge_to_load > 0 else 0
            soc -= raw_discharge
            soc = max(soc, usable_capacity * min_soc)
            total_discharge += batt_discharge_to_load

            # --- Battery Charge (from surplus PV) ---
            surplus_ac = max(0, e_use_ac - pv_to_load_ac)
            raw_charge = min(surplus_ac, pcs_capacity, max_charge_possible)
            useful_charge = raw_charge * pcs_eff * charge_eff
            soc += useful_charge

            # --- Import ---
            import_energy = max(0, load - pv_to_load_ac - batt_discharge_to_load)

            # --- Export & Excess ---
            pv_after_load_charge = max(0, e_use_ac - pv_to_load_ac - raw_charge)
            export = min(pv_after_load_charge, export_limit)
            excess = max(0, pv_after_load_charge - export)

            # --- Accumulate ---
            pv_to_load_total += pv_to_load_ac
            batt_use_total += batt_discharge_to_load
            import_total += min(import_energy, import_limit)
            export_total += export
            excess_total += excess
            pv_use_on_site = (pv_to_load_total / pv_prod_tot)*100
            ren_frac = ((pv_to_load_total + batt_use_total) / pv_prod_tot ) *100



        return {
            "DC Size (kW)": dc_size,
            "Inverter (kW)": inverter_capacity,
            "Battery (kWh)": battery_capacity,
            "PCS (kW)": pcs_capacity,
            "Load (kWh)": df_sim["Load"].sum(),
            "PV Production (kWh)": pv_prod_tot,
            "Solar→Load (kWh)": pv_to_load_total,
            "Battery Use (kWh)": batt_use_total,
            "Import (kWh)": import_total,
            "Export (kWh)": export_total,
            "Excess (kWh)": excess_total,
            "PV Used on site (%)": pv_use_on_site,
            "Pv and Battery by Produced (%)": ren_frac
        }


    # --- Batch Simulation UI ---
    st.header("📑 Batch Simulation")

    num_systems = st.number_input("How many systems to simulate?", min_value=1, max_value=20, value=2, step=1)

    batch_configs = []
    for i in range(num_systems):
        st.subheader(f"System {i + 1}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dc = st.number_input(f"DC Size (kW) - Sys {i + 1}", value=40.0, key=f"dc_{i}")
            inv = st.number_input(f"Inverter (kW) - Sys {i + 1}", value=30.0, key=f"inv_{i}")
        with c2:
            batt = st.number_input(f"Battery (kWh) - Sys {i + 1}", value=50.0, key=f"batt_{i}")
            pcs = st.number_input(f"PCS (kW) - Sys {i + 1}", value=30.0, key=f"pcs_{i}")
        with c3:
            inv_eff = st.number_input(f"Inverter Eff (%) - Sys {i + 1}", value=98.0, key=f"inv_eff_{i}") / 100
            batt_eff = st.number_input(f"Battery Eff (%) - Sys {i + 1}", value=96.0, key=f"batt_eff_{i}") / 100
        with c4:
            pcs_eff = st.number_input(f"PCS Eff (%) - Sys {i + 1}", value=98.0, key=f"pcs_eff_{i}") / 100

        batch_configs.append((dc, inv, batt, pcs, inv_eff, batt_eff, pcs_eff))

    if st.button("▶ Run Batch Simulation"):
        results_list = []
        for cfg in batch_configs:
            res = simulate_one_system(cfg[0], cfg[1], cfg[2], cfg[3],
                                      cfg[4], cfg[5], cfg[6],
                                      base_dc_size, df, dod, min_soc,
                                      initial_soc, battery_qty, c_rate,
                                      export_limit, import_limit)
            results_list.append(res)

        res_df = pd.DataFrame(results_list)
        st.subheader("Batch Results (Annual Summary)")
        st.dataframe(res_df)

        st.download_button("📥 Download Batch Results",
                           res_df.to_csv(index=False),
                           "batch_results.csv",
                           "text/csv",
                           key="download_batch")

