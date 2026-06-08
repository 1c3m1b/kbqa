import executor.sparql_executor as se


def ensure_odbc():
    if se.odbc_conn is None:
        se.initialize_odbc_connection()
    se.odbc_conn.timeout = 30


def run(sql):
    ensure_odbc()
    print("\nSQL:")
    print(sql)
    try:
        with se.odbc_conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchmany(20)
    except Exception as e:
        print("[FAILED]", repr(e))
        return

    if not rows:
        print("[NO ROWS]")
        return

    for row in rows:
        print(tuple(row))


run("SELECT COUNT(*) FROM DB.DBA.RDF_QUAD")

run("SELECT COUNT(*) FROM DB.DBA.RDF_IRI")

run("""
SELECT COUNT(*)
FROM DB.DBA.RDF_IRI
WHERE RI_NAME LIKE 'http://rdf.freebase.com/ns/%'
""")

run("""
SELECT TOP 20 RI_ID, RI_NAME
FROM DB.DBA.RDF_IRI
WHERE RI_NAME LIKE 'http://rdf.freebase.com/ns/%'
""")

run("""
SELECT RI_ID, RI_NAME
FROM DB.DBA.RDF_IRI
WHERE RI_NAME = 'http://rdf.freebase.com/ns/m.02mjmr'
""")

run("""
SELECT RI_ID, RI_NAME
FROM DB.DBA.RDF_IRI
WHERE RI_NAME = 'http://rdf.freebase.com/ns/type.object.type'
""")

run("""
SELECT COUNT(*)
FROM DB.DBA.RDF_QUAD
WHERE P = IRI_TO_ID('http://rdf.freebase.com/ns/type.object.type')
""")

run("""
SELECT COUNT(*)
FROM DB.DBA.RDF_QUAD
WHERE S = IRI_TO_ID('http://rdf.freebase.com/ns/m.02mjmr')
""")