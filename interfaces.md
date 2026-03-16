c# Interface Matrix: UI / REPL / Jupyter

## Mechanisms

| Mechanism | Description |
|---|---|
| **REST** | Direct HTTP endpoint (`GET`/`POST`/`PUT`/`DELETE /api/...`) |
| **WS action** | WebSocket message (`{"action": "...", "data": {...}}`) sent to active session WS |
| **Command** | Slash command sent as query text via `POST /api/sessions/{id}/query` with `problem: "/command"`. Backend command dispatcher intercepts before normal pipeline. |
| **SSE** | Server-Sent Events stream (HTTP streaming response) |

## Feature Matrix

| Feature | Vera UI | REPL | Jupyter (`constat_jupyter`) | Mechanism |
|---|---|---|---|---|
| **Session Management** | | | | |
| Create session | Auto on load | Auto on start | `client.create_session()` | REST |
| List sessions | Session sidebar | `/history` | `client.list_sessions()` | REST |
| Resume session | Click session | `/resume <id>` | `client.get_session(id)` | REST |
| Delete session | Delete button | — | `client.delete_session(id)` | REST |
| Session status | Status bar | `/state` | `session.status` | REST |
| Reset session | — | `/reset` | `session.reset()` | REST |
| Context info | — | `/context` | `session.context()` | REST |
| Compact context | — | `/compact` | `session.compact()` | Command |
| Set user | — | `/user [name]` | via auth token | — |
| **Query Execution** | | | | |
| Ask question | Chat input | Type question | `session.solve(question)` | REST+WS |
| Follow-up | Chat input | Type question | `session.follow_up(question)` | REST+WS |
| Reason chain | Toggle auditable | `/reason` | `session.reason_chain(question)` | REST+WS |
| Audit last result | — | `/audit` | `session.audit()` | Command |
| Plan approval | Approval widget | Y/n prompt | `input()` prompt (`auto_approve=False`) | WS action |
| Clarification | Clarification widget | Input prompt | `input()` prompt | WS action |
| Cancel query | Cancel button | Ctrl+C | `session.cancel()` | REST |
| Redo query | — | `/redo [instruction]` | `session.redo(instruction)` | Command |
| **Plan Management** | | | | |
| View plan | Plan panel | Shown inline | `session.plan()` | REST |
| View steps | Steps panel | Shown inline | `session.steps()` | REST |
| Edit step | Inline edit | `/step-edit <N> <goal>` | `session.step_edit(n, goal)` | WS action |
| Delete step | Delete button | `/step-delete <N>` | `session.step_delete(n)` | WS action |
| Redo from step | — | `/step-redo <N>` | `session.step_redo(n)` | WS action |
| Save plan | — | `/save <name>` | `session.save_plan(name)` | Command |
| Share plan | — | `/share <name>` | `session.share_plan(name)` | Command |
| List plans | — | `/plans` | `session.list_plans()` | Command |
| Replay plan | — | `/replay <name>` | `session.replay_plan(name)` | Command |
| **Objectives** | | | | |
| List objectives | Objectives panel | `/objectives` | `session.objectives()` | Command |
| Edit objective | Inline edit | `/objective-edit <N> <text>` | `session.edit_objective(n, text)` | WS action |
| Delete objective | Delete button | `/objective-delete <N>` | `session.delete_objective(n)` | WS action |
| **Data Exploration** | | | | |
| List tables | Tables panel | `/tables` | `session.tables()` | REST |
| Show table | Click table | `/show <table>` | `session.table(name)` | REST |
| View code | Code panel | `/code [step]` | `session.code(step)` | REST |
| Download code | Download button | `/download-code [file]` | `session.download_code()` | REST |
| Inference codes | — | — | `session.inference_codes()` | REST |
| List artifacts | Artifacts panel | `/artifacts [all]` | `session.artifacts()` | REST |
| View artifact | Click artifact | — | `session.artifact(id).display()` | REST |
| Star table | Star button | — | `session.star_table(name)` | REST |
| Delete table | Delete button | — | `session.delete_table(name)` | REST |
| Star artifact | Star button | — | `session.star_artifact(id)` | REST |
| Delete artifact | Delete button | — | `session.delete_artifact(id)` | REST |
| Export table | Download button | `/export <table> [file]` | `session.export_table(name, path)` | REST |
| Summarize | — | `/summarize <target>` | `session.summarize(target)` | Command |
| DDL | — | — | `session.ddl()` | REST |
| Scratchpad | — | — | `session.scratchpad()` | REST |
| Output summary | — | — | `session.output()` | REST |
| **Data Sources** | | | | |
| List databases | Schema panel | `/databases` | `session.databases()` | REST |
| Add database | Add DB dialog | `/database <uri>` | `session.add_database(uri)` | REST |
| Remove database | — | — | `session.remove_database(name)` | REST |
| List all sources | Sources panel | — | `session.sources()` | REST |
| Add API | Add API dialog | `/api <spec_url>` | `session.add_api(spec_url)` | REST |
| Remove API | — | — | `session.remove_api(name)` | REST |
| Add document | Upload button | `/doc <path>` | `session.add_document(uri)` / `session.upload_document(path)` | REST |
| List files | Files panel | `/files` | `session.files()` | REST |
| Add file | Upload button | `/file <uri>` | `session.upload_file(path)` | REST |
| Delete file | Delete button | — | `session.delete_file(id)` | REST |
| Discover sources | — | `/discover <query>` | `session.discover(query)` | Command |
| **Diagnostics** | | | | |
| Table schema | Schema panel | `/schema <db.table>` | `client.table_schema(db, table)` | REST |
| Search schema | — | `/search-tables <query>` | `session.search_tables(query)` | REST |
| Search APIs | — | `/search-apis <query>` | `session.search_apis(query)` | REST |
| Search documents | — | `/search-docs <query>` | `session.search_docs(query)` | REST |
| Search chunks | — | `/search-chunks <query>` | `session.search_chunks(query)` | REST |
| Glossary lookup | Glossary panel | `/lookup <name>` | `session.glossary_term(name)` | REST |
| Entity search | Entity panel | `/entity <name>` | `session.entities()` | REST |
| Known facts | Facts panel | `/known-facts` | `session.facts()` | REST |
| Source search | — | `/sources <query>` | `client.search_schema(query)` | REST |
| Proof tree | — | — | `session.proof_tree()` | REST |
| **Glossary** | | | | |
| List terms | Glossary panel | `/glossary [scope]` | `session.glossary()` | REST |
| View term | Click term | `/lookup <name>` | `session.glossary_term(name)` | REST |
| Define term | Edit in panel | `/define <name> <def>` | `session.define(name, definition)` | REST |
| Remove term | Delete button | `/undefine <name>` | `session.undefine(name)` | REST |
| Refine term | AI button | `/refine <name>` | `session.refine(name)` | REST |
| Generate glossary | Generate button | — | `session.generate_glossary()` | REST |
| **Facts & Memory** | | | | |
| List facts | Facts panel | `/facts` | `session.facts()` | REST |
| Remember fact | — | `/remember <fact>` | `session.remember(name, value)` | REST |
| Forget fact | Delete button | `/forget <name>` | `session.forget(name)` | REST |
| Record correction | — | `/correct <text>` | `session.correct(text)` | REST |
| List learnings | — | `/learnings [cat]` | `client.learnings(category)` | REST |
| Compact learnings | — | `/compact-learnings` | `client.compact_learnings()` | REST |
| Add rule | — | `/rule <text>` | `client.add_rule(text)` | REST |
| Edit rule | — | `/rule-edit <id> <text>` | `client.edit_rule(id, text)` | REST |
| Delete rule | — | `/rule-delete <id>` | `client.delete_rule(id)` | REST |
| **Agents** | | | | |
| List agents | Agent selector | `/agents` | `session.agents()` | REST |
| Set agent | Dropdown | `/agent <name>` | `session.set_agent(name)` | REST |
| Get agent | — | — | `session.agent(name)` | REST |
| Create agent | — | `/agent-create <name>` | `session.create_agent(name, content)` | REST |
| Edit agent | — | `/agent-edit <name>` | `session.edit_agent(name, content)` | REST |
| Delete agent | — | `/agent-delete <name>` | `session.delete_agent(name)` | REST |
| Draft agent | — | `/agent-draft <name> <desc>` | `session.draft_agent(name, desc)` | REST |
| **Skills** | | | | |
| List skills | Skills panel | `/skills` | `client.skills()` | REST |
| Skill info | Click skill | `/skill <name>` | `client.skill_info(name)` | REST |
| Create skill | — | `/skill-create <name>` | `client.create_skill(name, content)` | REST |
| Edit skill | — | `/skill-edit <name>` | `client.edit_skill(name, content)` | REST |
| Delete skill | — | `/skill-delete <name>` | `client.delete_skill(name)` | REST |
| Draft skill | — | `/skill-draft <name> <desc>` | `client.draft_skill(name, desc)` | REST |
| Download skill | — | `/skill-download <name>` | `client.download_skill(name)` | REST |
| **Domains** | | | | |
| List domains | Domain sidebar | — | `client.domains()` | REST |
| Set domains | Domain selector | — | `session.set_domains([...])` | REST |
| **Regression Testing** | | | | |
| List test domains | Regression panel | `constat test` | `session.test_domains()` | REST |
| List questions | Regression panel | — | `session.test_questions(domain)` | REST |
| Create question | Add button | — | `session.create_test_question(domain, q)` | REST |
| Update question | Edit button | — | `session.update_test_question(domain, idx, q)` | REST |
| Delete question | Delete button | — | `session.delete_test_question(domain, idx)` | REST |
| Run tests | Run button | `constat test` | `session.run_tests(domains)` | SSE |
| **Display Options** | | | | |
| Verbose mode | — | `/verbose [on/off]` | `session.set_verbose(on)` | Command |
| Raw output | — | `/raw [on/off]` | `session.set_raw(on)` | Command |
| Insights toggle | — | `/insights [on/off]` | `session.set_insights(on)` | Command |
| Preferences | Settings | `/preferences` | `session.preferences()` | Command |
| **Results** | | | | |
| View answer | Chat output | Printed inline | `result.answer` / `result` | — |
| View tables | Tables panel | `/tables`, `/show` | `result.tables` | — |
| Display published | — | — | `result.display(published=True)` | — |
| Suggestions | Suggestion chips | Printed inline | `result.suggestions` | — |
| Messages | Chat history | — | `session.messages()` | REST |
| Any command | Chat input | Type `/command` | `session.command("/command args")` | Command |
