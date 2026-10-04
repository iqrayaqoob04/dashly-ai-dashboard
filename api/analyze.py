"""Allowlisted, deterministic data-analysis endpoint for Dashly."""
import base64, io, json, re
from http.server import BaseHTTPRequestHandler
import pandas as pd
import numpy as np

INTENTS={"row_count","count","percentage","prevalence","conditional_count","conditional_percentage","sum","average","median","min","max","group_comparison","distribution","correlation","trend","ranking","missing_values","unique_values"}
YES={"yes","y","true","1","positive"}; NO={"no","n","false","0","negative"}

def tokens(value):
    return [x for x in re.split(r"[^a-z0-9]+",str(value).lower()) if len(x)>1]

def normalized(s):
    return s.astype("string").str.strip().str.lower()

def similar(a,b):
    return a==b or (len(a)>=4 and len(b)>=4 and a[:4]==b[:4])

def profile(df):
    out=[]
    for col in df.columns:
        s=df[col]; non=s.dropna(); numeric=pd.to_numeric(non,errors="coerce")
        dates=pd.to_datetime(non,errors="coerce")
        kind="numeric" if len(non) and numeric.notna().mean()>=.8 else "date" if len(non) and dates.notna().mean()>=.8 else "categorical"
        out.append({"name":str(col),"kind":kind,"unique":int(s.nunique(dropna=True)),"missing_pct":round(float(s.isna().mean()*100),2),"samples":[str(x) for x in non.head(3).tolist()]})
    return out

def match_column(question, schema, kinds=None):
    q=tokens(question); ranked=[]
    for item in schema:
        if kinds and item["kind"] not in kinds: continue
        name=tokens(item["name"]); score=sum(4 for w in name if w in q)
        phrase=" ".join(name)
        if phrase and phrase in " ".join(q): score+=8
        # Category values (for example "male") can identify their column (GENDER).
        score+=sum(3 for sample in item.get("samples",[]) for w in tokens(sample) if w in q)
        if score: ranked.append((score,item["name"]))
    if not ranked:return None
    ranked.sort(reverse=True)
    if len(ranked)>1 and ranked[0][0]==ranked[1][0]: return None
    return ranked[0][1]

def matching_columns(question, schema, kinds):
    q=tokens(question); found=[]
    for item in schema:
        if item["kind"] not in kinds: continue
        name=tokens(item["name"]); matches=[i for i,w in enumerate(q) if any(similar(w,n) for n in name)]
        sample_matches=[i for i,w in enumerate(q) if any(similar(w,v) for sample in item.get("samples",[]) for v in tokens(sample))]
        if matches or sample_matches: found.append((min(matches+sample_matches),item["name"]))
    return [name for _,name in sorted(found)]

def values_for_condition(question, schema, column):
    samples=next(x["samples"] for x in schema if x["name"]==column)
    mentioned=[w for v in samples for w in tokens(v) if w in tokens(question)]
    return mentioned or list(YES)

def intent_for(q):
    q=q.lower()
    if re.search(r"how many|number of|total (patients|records|rows|customers|entries)",q) and not re.search(r"have |with |by ",q):return "row_count"
    if "%" in q or re.search(r"percentage|percent|prevalence",q):return "percentage"
    if "how many" in q or re.search(r"count .*?(have|with)",q):return "count"
    if re.search(r"average|mean",q):return "average"
    if "median" in q:return "median"
    if re.search(r"minimum|lowest|min ",q):return "min"
    if re.search(r"maximum|highest|max ",q):return "max"
    if re.search(r"sum|total sales|total revenue",q):return "sum"
    if re.search(r"compare|versus|\\bvs\\b",q):return "group_comparison"
    if re.search(r"trend|over time|change over time",q):return "trend"
    if "missing" in q:return "missing_values"
    if "unique|distinct" in q:return "unique_values"
    if "by " in q:return "distribution"
    return None

def plan(question,schema):
    intent=intent_for(question)
    if not intent: raise ValueError("I couldn't determine the analysis. Please ask for a count, percentage, average, comparison, trend, or grouping.")
    numeric=match_column(question,schema,{"numeric"}); categorical=match_column(question,schema,{"categorical"}); date=match_column(question,schema,{"date"})
    q=question.lower()
    # Conditions prefer fields named after words following have/with/of, never arbitrary numeric fields.
    condition=match_column(question,schema,{"categorical"})
    if intent in {"row_count"}: return {"intent":intent}
    if intent in {"count","percentage"}:
        conditions=matching_columns(question,schema,{"categorical"})
        if len(conditions)>=2:
            if re.search(r"due to|because of|caused by|causes",q):
                raise ValueError("I can measure the observed relationship between these variables, but this dataset cannot establish causation. Please clarify: do you want the percentage within the first group, or the percentage of all records that meet both conditions?")
            condition_specs=[{"column":col,"values":values_for_condition(question,schema,col)} for col in conditions]
            denominator=[] if re.search(r"all (patients|records|customers)|entire dataset|all rows",q) else [condition_specs[0]]
            return {"intent":"conditional_percentage" if intent=="percentage" else "conditional_count","numerator_conditions":condition_specs,"denominator_conditions":denominator}
        if not condition: raise ValueError("I couldn't determine which category or condition to count. Please name it, for example 'patients with lung cancer'.")
        positives=values_for_condition(question,schema,condition)
        return {"intent":"prevalence" if intent=="percentage" else "count","condition_column":condition,"positive_values":positives}
    if intent in {"average","median","sum","min","max"}:
        if not numeric: raise ValueError("I couldn't determine the numeric column. Please specify the metric, for example 'average age'.")
        return {"intent":intent,"metric_column":numeric}
    if intent=="group_comparison":
        if not numeric or not categorical: raise ValueError("Please name both a numeric metric and a group, for example 'compare income by region'.")
        return {"intent":intent,"metric_column":numeric,"group_column":categorical}
    if intent=="trend":
        if not numeric or not date: raise ValueError("A trend needs a numeric metric and a date column. Please specify both.")
        return {"intent":intent,"metric_column":numeric,"date_column":date}
    if intent=="distribution":
        if not categorical: raise ValueError("I couldn't determine the category to group by. Please name the column.")
        return {"intent":intent,"group_column":categorical}
    if intent=="missing_values": return {"intent":intent}
    if intent=="unique_values":
        col=match_column(question,schema)
        if not col: raise ValueError("Please specify the column whose unique values you want.")
        return {"intent":intent,"column":col}
    raise ValueError("That analysis is not supported.")

def condition_mask(df,col,positive_values=YES):
    values=normalized(df[col]); positives=values.isin(set(positive_values))
    # If no Yes-like encoding, use explicitly mentioned category value when it uniquely occurs.
    return positives, int(values.notna().sum())

def conditions_mask(df, conditions):
    mask=pd.Series(True,index=df.index)
    for condition in conditions:
        col=condition["column"]
        if col not in df.columns: raise ValueError(f"Referenced column {col} is not in the dataset.")
        values=set(condition["values"])
        current=normalized(df[col]).isin(values)
        if not current.any(): raise ValueError(f"No values matching {col} = {condition['values'][0]} were found.")
        mask &= current
    return mask

def execute(df,p):
    intent=p["intent"]; n=len(df)
    if not n: raise ValueError("The dataset has no rows.")
    result={"intent":intent,"plan":p,"kpis":[],"chart":None,"provenance":[]}
    if intent=="row_count":
        result["answer"]=f"There are {n:,} records in the dataset."
        result["kpis"]=[["Dataset rows",n,"Calculated with len(df)"]]; result["provenance"]=[f"Calculated from {n:,} dataset rows using len(df)."]
    elif intent in {"conditional_count","conditional_percentage"}:
        numerator=conditions_mask(df,p["numerator_conditions"])
        denominator=conditions_mask(df,p["denominator_conditions"]) if p["denominator_conditions"] else pd.Series(True,index=df.index)
        numerator_count,denominator_count=int(numerator.sum()),int(denominator.sum())
        if denominator_count==0 or numerator_count>denominator_count or (numerator & ~denominator).any(): raise ValueError("Conditional-analysis sanity check failed.")
        percentage=numerator_count/denominator_count*100
        labels=[" AND ".join(f"{c['column']} = {c['values'][0]}" for c in p["numerator_conditions"])]
        scope="all dataset rows" if not p["denominator_conditions"] else " AND ".join(f"{c['column']} = {c['values'][0]}" for c in p["denominator_conditions"])
        result["kpis"]=[["Matching records",numerator_count,labels[0]],["Denominator",denominator_count,scope],["Percentage",f"{percentage:.2f}%","Matching records ÷ denominator"]]
        result["answer"]=f"{numerator_count:,} of {denominator_count:,} records meet both conditions, representing {percentage:.2f}%." if intent=="conditional_percentage" else f"{numerator_count:,} records meet both conditions."
        result["chart"]={"type":"bar","title":"Conditional analysis result","labels":["Meet both conditions","Denominator only"],"values":[numerator_count,denominator_count-numerator_count]}
        result["provenance"]=[f"Dataset rows: {n:,}.",f"Numerator: {labels[0]}.",f"Denominator: {scope}.",f"Formula: {numerator_count} ÷ {denominator_count} × 100 = {percentage:.2f}%.", "This is an observed relationship and does not establish causation."]
    elif intent in {"count","prevalence"}:
        col=p["condition_column"]; mask,valid=condition_mask(df,col,p.get("positive_values",YES)); count=int(mask.sum())
        if valid==0: raise ValueError(f"{col} has no usable values.")
        pct=count/n*100
        if not 0<=pct<=100 or count>n: raise ValueError("Sanity check failed for the condition count.")
        label=f"{col} = {p.get('positive_values',['Yes'])[0]}"
        result["kpis"]=[["Matching records",count,label],["Dataset rows",n,"All rows"],["Percentage",f"{pct:.2f}%","Matching records ÷ all rows"]]
        result["answer"]=f"{count:,} of {n:,} records have {label}, representing {pct:.2f}% of the dataset." if intent=="prevalence" else f"{count:,} of {n:,} records have {label}."
        yes,no=count,n-count; result["chart"]={"type":"bar","title":f"{col}: Yes vs other","labels":["Yes","Other / missing"],"values":[yes,no]}
        result["provenance"]=[f"Calculated from {n:,} dataset rows using {col} normalized to Yes/No values."]
    elif intent in {"average","median","sum","min","max"}:
        col=p["metric_column"]; s=pd.to_numeric(df[col],errors="coerce").dropna()
        if s.empty: raise ValueError(f"{col} is not usable as a numeric field.")
        value={"average":s.mean(),"median":s.median(),"sum":s.sum(),"min":s.min(),"max":s.max()}[intent]
        if intent=="average" and not s.min()<=value<=s.max(): raise ValueError("Sanity check failed for the mean.")
        label={"average":"Average","median":"Median","sum":"Total","min":"Minimum","max":"Maximum"}[intent]
        result["answer"]=f"{label} {col}: {value:,.2f}."
        result["kpis"]=[[label+" "+col,f"{value:,.2f}",f"{len(s):,} valid values"],["Missing",int(df[col].isna().sum()),"Excluded from calculation"]]
        result["provenance"]=[f"Calculated from {col}, excluding {int(df[col].isna().sum()):,} missing values."]
    elif intent=="group_comparison":
        m,g=p["metric_column"],p["group_column"]; work=df[[m,g]].copy();work[m]=pd.to_numeric(work[m],errors="coerce");work[g]=normalized(work[g]).str.title();work=work.dropna()
        if work.empty: raise ValueError("No valid numeric values are available for this comparison.")
        grouped=work.groupby(g,dropna=False)[m].mean().sort_values(ascending=False).head(12)
        result["answer"]=f"{m} is highest for {grouped.index[0]} ({grouped.iloc[0]:,.2f})."
        result["kpis"]=[["Highest average",f"{grouped.iloc[0]:,.2f}",str(grouped.index[0])],["Groups compared",len(grouped),g]]
        result["chart"]={"type":"bar","title":f"Average {m} by {g}","labels":[str(x) for x in grouped.index],"values":[round(float(x),4) for x in grouped.values]}
        result["provenance"]=[f"Grouped by {g} and averaged {m}; rows with missing values were excluded."]
    elif intent=="distribution":
        g=p["group_column"]; counts=df[g].fillna("Missing").astype(str).value_counts().head(12)
        result["answer"]=f"Distribution of records by {g}."
        result["kpis"]=[["Categories",len(counts),g],["Largest group",int(counts.iloc[0]),str(counts.index[0])]]
        result["chart"]={"type":"bar","title":f"Records by {g}","labels":counts.index.tolist(),"values":counts.astype(int).tolist()}
        result["provenance"]=[f"Counted all {n:,} rows grouped by {g}; missing values are shown separately."]
    elif intent=="missing_values":
        missing=df.isna().sum().sort_values(ascending=False).head(12)
        result["answer"]="Missing-value profile for the dataset."
        result["kpis"]=[["Dataset rows",n,"All rows"],["Most missing",int(missing.iloc[0]),str(missing.index[0])]]
        result["chart"]={"type":"bar","title":"Missing values by column","labels":missing.index.tolist(),"values":missing.astype(int).tolist()}
        result["provenance"]=["Counted null values in every column."]
    elif intent=="unique_values":
        col=p["column"]; value=int(df[col].nunique(dropna=True))
        result["answer"]=f"{col} has {value:,} unique non-missing values."
        result["kpis"]=[["Unique values",value,col]];result["provenance"]=[f"Calculated unique non-missing values from {col}."]
    return result

def analyze(file_bytes, filename, question):
    try:
        df=pd.read_csv(io.BytesIO(file_bytes)) if filename.lower().endswith(".csv") else pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e: raise ValueError("I couldn't read that file. Please upload a valid CSV, XLSX, or XLS file.") from e
    df.columns=[str(c).strip() for c in df.columns]
    schema=profile(df); p=plan(question,schema); out=execute(df,p);out["schema"]=schema;return out

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length=int(self.headers.get("Content-Length",0))
            if length>4_000_000: raise ValueError("This deployment accepts files up to 4 MB. Please use a smaller file.")
            body=json.loads(self.rfile.read(length)); raw=base64.b64decode(body["file"]); out=analyze(raw,body["filename"],body["question"])
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(json.dumps(out,default=str).encode())
        except ValueError as e:self.send_response(400);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(json.dumps({"error":str(e)}).encode())
        except Exception:self.send_response(500);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(json.dumps({"error":"Analysis failed safely. Please try another file or question."}).encode())