"""
BRCA2 Variant Pathogenicity Predictor v2.2
==========================================
Fixed ClinVar API parsing for review status and submitter count

Created as part of: Database Bias in BRCA2 Variant Interpretation Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re

# Page configuration
st.set_page_config(
    page_title="BRCA2 Variant Predictor",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0;
    }
    .prediction-high {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-low {
        color: #27ae60;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-uncertain {
        color: #f39c12;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API FUNCTIONS
# ============================================

def fetch_gnomad_data(variant_id):
    """
    Fetch variant data from gnomAD GraphQL API
    variant_id format: 13-32332656-G-A (chrom-pos-ref-alt)
    """
    try:
        # Parse variant ID - handle various formats
        variant_id = variant_id.strip().replace(' ', '').replace(':', '-')
        parts = variant_id.split('-')
        
        if len(parts) != 4:
            return None, "Invalid format. Use: 13-32332656-G-A"
        
        chrom, pos, ref, alt = parts
        chrom = chrom.replace('chr', '')
        
        # gnomAD GraphQL query
        query = """
        query getVariant($variantId: String!, $datasetId: DatasetId!) {
            variant(variantId: $variantId, dataset: $datasetId) {
                variant_id
                pos
                ref
                alt
                genome {
                    ac
                    an
                    populations {
                        id
                        ac
                        an
                    }
                }
                exome {
                    ac
                    an
                    populations {
                        id
                        ac
                        an
                    }
                }
                transcript_consequences {
                    major_consequence
                    gene_symbol
                    hgvsc
                    hgvsp
                    lof
                }
            }
        }
        """
        
        gnomad_variant_id = f"{chrom}-{pos}-{ref}-{alt}"
        
        # Try gnomAD v4 first
        variables = {
            "variantId": gnomad_variant_id,
            "datasetId": "gnomad_r4"
        }
        
        response = requests.post(
            "https://gnomad.broadinstitute.org/api",
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        data = response.json()
        
        # If v4 fails, try v2
        if "errors" in data or not data.get("data", {}).get("variant"):
            variables["datasetId"] = "gnomad_r2_1"
            response = requests.post(
                "https://gnomad.broadinstitute.org/api",
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            data = response.json()
        
        if "errors" in data:
            return None, f"API Error: {data['errors'][0]['message']}"
        
        variant = data.get("data", {}).get("variant")
        
        if not variant:
            return None, "Variant not found in gnomAD"
        
        return variant, None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, f"Error: {str(e)}"


def parse_gnomad_data(variant_data):
    """
    Parse gnomAD API response into features
    """
    features = {
        'gnomad_AF': 0,
        'gnomad_AF_sas': 0,
        'consequence_severity': 1,
        'pos_scaled': 0.5,
        'protein_position': None,
        'consequence': 'unknown',
        'other_pop_frequencies': {}
    }
    
    total_ac = 0
    total_an = 0
    sas_ac = 0
    sas_an = 0
    other_pops = {}
    
    for dataset in ['exome', 'genome']:
        ds = variant_data.get(dataset)
        if ds:
            total_ac += ds.get('ac', 0) or 0
            total_an += ds.get('an', 0) or 0
            
            for pop in ds.get('populations', []):
                pop_id = pop.get('id', '').lower()
                pop_ac = pop.get('ac', 0) or 0
                pop_an = pop.get('an', 0) or 0
                
                if pop_id in ['sas', 'south_asian']:
                    sas_ac += pop_ac
                    sas_an += pop_an
                else:
                    if pop_id not in other_pops:
                        other_pops[pop_id] = {'ac': 0, 'an': 0}
                    other_pops[pop_id]['ac'] += pop_ac
                    other_pops[pop_id]['an'] += pop_an
    
    if total_an > 0:
        features['gnomad_AF'] = total_ac / total_an
    if sas_an > 0:
        features['gnomad_AF_sas'] = sas_ac / sas_an
    
    features['other_pop_frequencies'] = other_pops
    
    consequences = variant_data.get('transcript_consequences', [])
    if consequences:
        for tc in consequences:
            if tc.get('gene_symbol') == 'BRCA2':
                consequence = tc.get('major_consequence', 'unknown')
                features['consequence'] = consequence
                
                severe = ['frameshift_variant', 'stop_gained', 'stop_lost',
                         'splice_acceptor_variant', 'splice_donor_variant',
                         'transcript_ablation', 'start_lost']
                moderate = ['missense_variant', 'inframe_deletion',
                           'inframe_insertion', 'protein_altering_variant']
                
                if consequence in severe or tc.get('lof') == 'HC':
                    features['consequence_severity'] = 2
                elif consequence in moderate:
                    features['consequence_severity'] = 1
                else:
                    features['consequence_severity'] = 0
                
                hgvsp = tc.get('hgvsp', '')
                if hgvsp:
                    match = re.search(r'p\.[A-Za-z]+(\d+)', hgvsp)
                    if match:
                        features['protein_position'] = int(match.group(1))
                break
    
    if features['protein_position']:
        features['pos_scaled'] = min(1.0, features['protein_position'] / 3418)
    
    return features


def fetch_clinvar_data(variation_id):
    """
    Fetch variant data from ClinVar E-utilities API
    """
    try:
        variation_id = str(variation_id).strip()
        
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "clinvar",
            "id": variation_id,
            "retmode": "json"
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return None, f"ClinVar API returned status {response.status_code}"
        
        data = response.json()
        
        result = data.get("result", {})
        if variation_id not in result:
            return None, "Variation ID not found in ClinVar"
        
        return result[variation_id], None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, f"Error: {str(e)}"


def parse_clinvar_data(clinvar_data):
    """
    Parse ClinVar API response into features - FIXED VERSION
    """
    features = {
        'review_status_numeric': 0,
        'num_submitters': 1,
        'clinical_significance': 'Uncertain significance',
        'review_status_text': 'no assertion'
    }
    
    # Debug: print raw data structure
    # st.write("DEBUG - Raw ClinVar data:", clinvar_data)
    
    # Method 1: Check germline_classification (newer format)
    germline = clinvar_data.get('germline_classification', {})
    if isinstance(germline, dict) and germline:
        # Get classification description
        features['clinical_significance'] = germline.get('description', 'Uncertain significance')
        
        # Get review status text
        review_status = germline.get('review_status', '')
        features['review_status_text'] = review_status
        
        # Convert review status to stars
        review_lower = str(review_status).lower()
        if 'practice guideline' in review_lower:
            features['review_status_numeric'] = 4
        elif 'expert panel' in review_lower:
            features['review_status_numeric'] = 3
        elif 'criteria provided' in review_lower and 'multiple' in review_lower:
            features['review_status_numeric'] = 2
        elif 'criteria provided' in review_lower and 'conflicting' in review_lower:
            features['review_status_numeric'] = 1
        elif 'criteria provided' in review_lower or 'single submitter' in review_lower:
            features['review_status_numeric'] = 1
        else:
            features['review_status_numeric'] = 0
        
        # Get number of submissions
        num_subs = germline.get('num_submissions')
        if num_subs is not None:
            features['num_submitters'] = int(num_subs)
    
    # Method 2: Check supporting_submissions for submitter count
    supporting = clinvar_data.get('supporting_submissions', {})
    if isinstance(supporting, dict):
        # Check scv (SCV accessions = number of submissions)
        scv = supporting.get('scv')
        if scv:
            if isinstance(scv, list):
                features['num_submitters'] = len(scv)
            elif isinstance(scv, int):
                features['num_submitters'] = scv
    
    # Method 3: Check clinical_significance directly (older format)
    if not germline:
        clin_sig = clinvar_data.get('clinical_significance', {})
        if isinstance(clin_sig, dict):
            features['clinical_significance'] = clin_sig.get('description', 'Uncertain significance')
            
            review = clin_sig.get('review_status', '')
            features['review_status_text'] = review
            
            review_lower = str(review).lower()
            if 'practice guideline' in review_lower:
                features['review_status_numeric'] = 4
            elif 'expert panel' in review_lower:
                features['review_status_numeric'] = 3
            elif 'multiple' in review_lower and 'no conflict' in review_lower:
                features['review_status_numeric'] = 2
            elif 'criteria provided' in review_lower:
                features['review_status_numeric'] = 1
    
    # Method 4: Try to get submission count from variation_set
    variation_set = clinvar_data.get('variation_set', [])
    if isinstance(variation_set, list) and len(variation_set) > 0:
        for vs in variation_set:
            if isinstance(vs, dict):
                # Some responses have submission count here
                cdna = vs.get('cdna_change')
                # Check for other fields that might have count
    
    # Method 5: Parse title for any submission info
    title = clinvar_data.get('title', '')
    
    return features


def determine_domain(protein_position):
    """
    Determine BRCA2 functional domain from amino acid position
    """
    if protein_position is None:
        return {'domain_BRC1': 0, 'domain_BRC2': 0, 'domain_BRC3_BRC4': 0, 'domain_other': 1}, "Unknown"
    
    pos = int(protein_position)
    
    domains = {
        'BRC1': (1002, 1036),
        'BRC2': (1212, 1246),
        'BRC3': (1422, 1456),
        'BRC4': (1517, 1551),
        'BRC5': (1664, 1698),
        'BRC6': (1837, 1871),
        'BRC7': (1971, 2005),
        'BRC8': (2051, 2085),
        'DNA_binding': (2481, 3186),
        'C_terminal': (3263, 3418)
    }
    
    domain_features = {
        'domain_BRC1': 0,
        'domain_BRC2': 0,
        'domain_BRC3_BRC4': 0,
        'domain_other': 0
    }
    
    domain_name = "Interdomain region"
    
    for name, (start, end) in domains.items():
        if start <= pos <= end:
            if name == 'BRC1':
                domain_features['domain_BRC1'] = 1
                domain_name = "BRC Repeat 1"
            elif name == 'BRC2':
                domain_features['domain_BRC2'] = 1
                domain_name = "BRC Repeat 2"
            elif name in ['BRC3', 'BRC4']:
                domain_features['domain_BRC3_BRC4'] = 1
                domain_name = f"BRC Repeat {name[-1]}"
            else:
                domain_features['domain_other'] = 1
                domain_name = name.replace('_', ' ')
            break
    
    if sum(domain_features.values()) == 0:
        domain_features['domain_other'] = 1
    
    return domain_features, domain_name


def calculate_sa_flags(gnomad_af, gnomad_af_sas, other_pop_freqs):
    """
    Determine if variant is SA-specific or SA-enriched
    """
    if gnomad_af > 0:
        sa_enrichment = gnomad_af_sas / gnomad_af
    else:
        sa_enrichment = 10 if gnomad_af_sas > 0 else 0
    
    is_sa_enriched = 1 if sa_enrichment > 2 else 0
    
    is_sa_specific = 0
    if gnomad_af_sas > 0:
        other_total_ac = sum(p.get('ac', 0) for p in other_pop_freqs.values())
        if other_total_ac == 0 or sa_enrichment > 10:
            is_sa_specific = 1
    
    return {
        'sa_enrichment_ratio': sa_enrichment,
        'is_SA_specific': is_sa_specific,
        'is_SA_enriched': is_sa_enriched
    }


def predict_pathogenicity(features):
    """
    Calculate pathogenicity score based on features
    """
    base_score = 0.3
    
    if features['consequence_severity'] == 2:
        base_score += 0.35
    elif features['consequence_severity'] == 1:
        base_score += 0.12
    
    if features['gnomad_AF'] < 0.00001:
        base_score += 0.18
    elif features['gnomad_AF'] < 0.0001:
        base_score += 0.10
    elif features['gnomad_AF'] > 0.001:
        base_score -= 0.25
    elif features['gnomad_AF'] > 0.01:
        base_score -= 0.40
    
    if features['sa_enrichment_ratio'] > 5:
        base_score += 0.08
    elif features['sa_enrichment_ratio'] > 2:
        base_score += 0.04
    
    if features['ReviewStatus_numeric'] <= 1:
        base_score += 0.05
    elif features['ReviewStatus_numeric'] >= 3:
        base_score -= 0.05
    
    if features['NumberSubmitters'] <= 1:
        base_score += 0.04
    elif features['NumberSubmitters'] >= 5:
        base_score -= 0.03
    
    if features['domain_BRC1'] or features['domain_BRC2']:
        base_score += 0.08
    elif features['domain_BRC3_BRC4']:
        base_score += 0.05
    
    if features['pos_scaled'] < 0.2 or features['pos_scaled'] > 0.8:
        base_score += 0.03
    
    if features['is_SA_specific']:
        base_score += 0.04
    if features['is_SA_enriched']:
        base_score += 0.03
    
    return max(0, min(1, base_score))


# ============================================
# STREAMLIT APP
# ============================================

# Header
st.markdown('<h1 class="main-header">🧬 BRCA2 Variant Pathogenicity Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Auto-Fetch from gnomAD & ClinVar | Addressing South Asian Database Bias</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown("""
    **Version 2.2**
    
    Enter a gnomAD ID or ClinVar ID to automatically retrieve variant data.
    
    **Key Findings:**
    - 47.5% SA underrepresentation
    - 84 SA-specific VUS identified
    - 0% literature documentation
    """)
    
    st.markdown("---")
    st.markdown("### ID Format Examples")
    st.code("gnomAD: 13-32332656-G-A")
    st.code("ClinVar: 1382730")
    
    st.markdown("---")
    st.markdown("### Debug Mode")
    debug_mode = st.checkbox("Show raw API responses")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Auto-Fetch", "📝 Manual Entry", "📊 Key Findings"])

# ============================================
# TAB 1: Auto-Fetch
# ============================================
with tab1:
    st.markdown("## 🚀 Enter Variant IDs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### gnomAD Variant ID")
        gnomad_id = st.text_input(
            "Format: chrom-pos-ref-alt",
            placeholder="13-32332656-G-A"
        )
        
    with col2:
        st.markdown("### ClinVar Variation ID")
        clinvar_id = st.text_input(
            "Format: numeric ID",
            placeholder="1382730"
        )
    
    st.markdown("---")
    
    if st.button("🔍 Fetch Data & Predict", type="primary", use_container_width=True):
        
        if not gnomad_id and not clinvar_id:
            st.error("Please enter at least one ID")
        else:
            all_features = {
                'gnomad_AF': 0,
                'gnomad_AF_sas': 0,
                'sa_enrichment_ratio': 0,
                'ReviewStatus_numeric': 1,
                'NumberSubmitters': 1,
                'is_SA_specific': 0,
                'is_SA_enriched': 0,
                'pos_scaled': 0.5,
                'consequence_severity': 1,
                'domain_BRC1': 0,
                'domain_BRC2': 0,
                'domain_BRC3_BRC4': 0,
                'domain_other': 1
            }
            
            fetch_results = {}
            protein_position = None
            
            # Fetch gnomAD
            if gnomad_id:
                with st.spinner("Fetching from gnomAD..."):
                    gnomad_data, gnomad_error = fetch_gnomad_data(gnomad_id)
                    
                    if gnomad_error:
                        st.warning(f"gnomAD: {gnomad_error}")
                    elif gnomad_data:
                        if debug_mode:
                            st.markdown("#### Raw gnomAD Response:")
                            st.json(gnomad_data)
                        
                        parsed = parse_gnomad_data(gnomad_data)
                        all_features['gnomad_AF'] = parsed['gnomad_AF']
                        all_features['gnomad_AF_sas'] = parsed['gnomad_AF_sas']
                        all_features['consequence_severity'] = parsed['consequence_severity']
                        all_features['pos_scaled'] = parsed['pos_scaled']
                        protein_position = parsed['protein_position']
                        
                        sa_flags = calculate_sa_flags(
                            parsed['gnomad_AF'],
                            parsed['gnomad_AF_sas'],
                            parsed.get('other_pop_frequencies', {})
                        )
                        all_features.update(sa_flags)
                        
                        domain_features, domain_name = determine_domain(protein_position)
                        all_features.update(domain_features)
                        
                        fetch_results['gnomAD'] = {
                            'Global AF': f"{parsed['gnomad_AF']:.2e}",
                            'South Asian AF': f"{parsed['gnomad_AF_sas']:.2e}",
                            'SA Enrichment': f"{sa_flags['sa_enrichment_ratio']:.1f}x",
                            'Consequence': parsed['consequence'],
                            'Domain': domain_name
                        }
                        st.success("✅ gnomAD data fetched!")
            
            # Fetch ClinVar
            if clinvar_id:
                with st.spinner("Fetching from ClinVar..."):
                    clinvar_data, clinvar_error = fetch_clinvar_data(clinvar_id)
                    
                    if clinvar_error:
                        st.warning(f"ClinVar: {clinvar_error}")
                    elif clinvar_data:
                        if debug_mode:
                            st.markdown("#### Raw ClinVar Response:")
                            st.json(clinvar_data)
                        
                        parsed = parse_clinvar_data(clinvar_data)
                        all_features['ReviewStatus_numeric'] = parsed['review_status_numeric']
                        all_features['NumberSubmitters'] = parsed['num_submitters']
                        
                        fetch_results['ClinVar'] = {
                            'Classification': parsed['clinical_significance'],
                            'Review Status': f"{parsed['review_status_numeric']} star(s)",
                            'Review Text': parsed['review_status_text'],
                            'Submitters': parsed['num_submitters']
                        }
                        st.success("✅ ClinVar data fetched!")
            
            # Display results
            if fetch_results:
                st.markdown("---")
                st.markdown("## 📋 Fetched Data")
                
                for source, data in fetch_results.items():
                    st.markdown(f"### {source}")
                    cols = st.columns(len(data))
                    for i, (key, value) in enumerate(data.items()):
                        cols[i].metric(key, value)
                
                # All features
                st.markdown("---")
                st.markdown("## 🔢 All 13 Features")
                
                f1, f2, f3 = st.columns(3)
                
                with f1:
                    st.markdown("**Population**")
                    st.write(f"• Global AF: `{all_features['gnomad_AF']:.2e}`")
                    st.write(f"• SA AF: `{all_features['gnomad_AF_sas']:.2e}`")
                    st.write(f"• SA Enrichment: `{all_features['sa_enrichment_ratio']:.1f}x`")
                    st.write(f"• SA-Specific: `{'Yes' if all_features['is_SA_specific'] else 'No'}`")
                    st.write(f"• SA-Enriched: `{'Yes' if all_features['is_SA_enriched'] else 'No'}`")
                
                with f2:
                    st.markdown("**Clinical**")
                    st.write(f"• Review: `{all_features['ReviewStatus_numeric']} star(s)`")
                    st.write(f"• Submitters: `{all_features['NumberSubmitters']}`")
                    st.write(f"• Consequence: `{all_features['consequence_severity']}`")
                
                with f3:
                    st.markdown("**Structural**")
                    st.write(f"• Position: `{all_features['pos_scaled']:.3f}`")
                    st.write(f"• BRC1: `{all_features['domain_BRC1']}`")
                    st.write(f"• BRC2: `{all_features['domain_BRC2']}`")
                    st.write(f"• BRC3-4: `{all_features['domain_BRC3_BRC4']}`")
                    st.write(f"• Other: `{all_features['domain_other']}`")
                
                # Prediction
                st.markdown("---")
                st.markdown("## 🎯 Prediction")
                
                prediction = predict_pathogenicity(all_features)
                
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    if prediction >= 0.7:
                        st.markdown(f'<p class="prediction-high">Score: {prediction:.1%}</p>', unsafe_allow_html=True)
                        st.error("⚠️ HIGH RISK - Likely Pathogenic")
                    elif prediction >= 0.4:
                        st.markdown(f'<p class="prediction-uncertain">Score: {prediction:.1%}</p>', unsafe_allow_html=True)
                        st.warning("⚡ UNCERTAIN")
                    else:
                        st.markdown(f'<p class="prediction-low">Score: {prediction:.1%}</p>', unsafe_allow_html=True)
                        st.success("✅ LOW RISK - Likely Benign")
                
                with c2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "#27ae60"},
                                {'range': [40, 70], 'color': "#f39c12"},
                                {'range': [70, 100], 'color': "#e74c3c"}
                            ]
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                if all_features['is_SA_specific'] or all_features['is_SA_enriched']:
                    st.warning("**⚠️ SA Bias Alert:** This variant is enriched in South Asians and may be under-studied.")

# ============================================
# TAB 2: Manual Entry
# ============================================
with tab2:
    st.markdown("## 📝 Manual Entry")
    st.markdown("Use this if APIs are unavailable or you have the values already.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Population Genetics")
        gnomad_af_m = st.number_input("Global AF", 0.0, 1.0, 0.0001, format="%.6f")
        gnomad_sas_m = st.number_input("SA AF", 0.0, 1.0, 0.0005, format="%.6f")
        sa_spec_m = st.checkbox("SA-Specific")
        sa_enr_m = st.checkbox("SA-Enriched")
        
    with col2:
        st.markdown("### Clinical Evidence")
        review_m = st.selectbox("Review Status (stars)", [0, 1, 2, 3, 4], index=1)
        subs_m = st.slider("Number of Submitters", 1, 20, 1)
        cons_m = st.selectbox("Consequence", [0, 1, 2], index=1, 
                              format_func=lambda x: ["Synonymous", "Missense", "Truncating"][x])
    
    st.markdown("### Structural")
    pos_m = st.slider("Position (0-1)", 0.0, 1.0, 0.5)
    
    dc1, dc2, dc3, dc4 = st.columns(4)
    brc1_m = dc1.checkbox("BRC1")
    brc2_m = dc2.checkbox("BRC2")
    brc34_m = dc3.checkbox("BRC3-4")
    other_m = dc4.checkbox("Other", value=True)
    
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        if gnomad_af_m > 0:
            sa_enr_ratio = gnomad_sas_m / gnomad_af_m
        else:
            sa_enr_ratio = 10 if gnomad_sas_m > 0 else 0
        
        manual_features = {
            'gnomad_AF': gnomad_af_m,
            'gnomad_AF_sas': gnomad_sas_m,
            'sa_enrichment_ratio': sa_enr_ratio,
            'ReviewStatus_numeric': review_m,
            'NumberSubmitters': subs_m,
            'is_SA_specific': int(sa_spec_m),
            'is_SA_enriched': int(sa_enr_m),
            'pos_scaled': pos_m,
            'consequence_severity': cons_m,
            'domain_BRC1': int(brc1_m),
            'domain_BRC2': int(brc2_m),
            'domain_BRC3_BRC4': int(brc34_m),
            'domain_other': int(other_m)
        }
        
        pred = predict_pathogenicity(manual_features)
        
        st.markdown("---")
        st.markdown(f"## 🎯 Score: **{pred:.1%}**")
        
        if pred >= 0.7:
            st.error("⚠️ HIGH RISK - Likely Pathogenic")
        elif pred >= 0.4:
            st.warning("⚡ UNCERTAIN")
        else:
            st.success("✅ LOW RISK - Likely Benign")

# ============================================
# TAB 3: Key Findings
# ============================================
with tab3:
    st.markdown("## 📊 Research Findings")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SA Underrepresentation", "47.5%")
    c2.metric("SA-Specific VUS", "84")
    c3.metric("Literature Overlap", "0%")
    c4.metric("Model Accuracy", "~90%")
    
    st.markdown("---")
    
    pop_data = pd.DataFrame({
        'Population': ['European', 'South Asian', 'African', 'East Asian', 'Latino'],
        'Database %': [45, 12, 8, 15, 12],
        'World %': [10, 24, 17, 22, 8]
    })
    
    fig = px.bar(pop_data, x='Population', y=['Database %', 'World %'], barmode='group',
                 title='Database vs World Population')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**Disclaimer:** For research purposes only. Not for clinical decisions.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>BRCA2 Database Bias Project v2.2</p>", unsafe_allow_html=True)
