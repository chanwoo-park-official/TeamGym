"""Financial scales must survive the PIU-only consumer without oracle hints."""
import hashlib

from teamgym.data.finance.finlongdocqa.source import extract_unit_conventions, index_report
from teamgym.data.finance.finlongdocqa.partition import build_partitioned_datapoint
from teamgym.data.finance.finlongdocqa.validation import hydrate_task
from teamgym.interaction.llm_actors import _public_task_payload
from teamgym.interaction.rui_context import rui_only_datapoint


def test_exact_declaration_excludes_nearby_answers_and_deduplicates_whitespace():
    note = 'Tabular amounts are in U.S. Dollars in thousands, except share amounts, unless otherwise noted.'
    text = '# Page 24\nRevenue was 999. '+note+' Earnings were 888.\n# Page 82\n'+note.replace(' ', '  ')
    rows = extract_unit_conventions(text)
    assert len(rows)==1
    row=rows[0]
    assert row['text']==note and row['page_number']==24
    assert text[row['raw_start']:row['raw_end']]==note
    assert row['sha256']==hashlib.sha256(note.encode()).hexdigest()
    assert '999' not in row['text'] and '888' not in row['text']


def test_absent_or_unqualified_scale_is_not_invented():
    assert extract_unit_conventions('Revenue was 496,547. The dollar value is unspecified.')==[]
    assert extract_unit_conventions('The table might be in thousands.')==[]


def test_remote_unit_convention_is_visible_in_both_runtime_conditions():
    note='Tabular amounts are in U.S. Dollars in thousands, unless otherwise noted.'
    text=('# Page 1\n'+note+'\n\n# Page 2\n\n| Year | Income |\n|---|---|\n|2022|496547|\n'
          '\n# Page 3\n\nTotal assets were 28.1 billion.')
    indexed=index_report('X_2022',text)
    table=next(b for b in indexed['blocks'] if b['block_type']=='table')
    assets=next(b for b in indexed['blocks'] if 'Total assets' in b['text'])
    task={'source_qa_ids_used':['q'], 'public_task':{'question':'Compute yield.','response_requirements':['Show calculation.']},
          'evidence_units':[{'unit_id':f'riu_{i}', 'source_block_id':b['block_id']} for i,b in enumerate([table,assets],1)],
          'rubric':{'required_claims':[{'claim_id':'c','description':'Yield','expected_answer_element':'1.77%',
                    'supporting_unit_ids':['riu_1','riu_2'],'grading_rule':'Use correct currency scale.'}]},
          'oracle':{'reference_answer':'496.547 million divided by 28.1 billion is about 1.77%.'}}
    task=hydrate_task(task,[table,assets])
    row=build_partitioned_datapoint(preserve_rui_order=True, report_id='X_2022',company='X',year='2022',report_source={},
        report_index=indexed,task=task,qa_mappings=[],solver_audit={})
    for view in [row,rui_only_datapoint(row)]:
        visible=_public_task_payload(view)['initial_common_information']
        assert visible['source_unit_conventions']==[note]
        assert '496547' not in str(visible)
