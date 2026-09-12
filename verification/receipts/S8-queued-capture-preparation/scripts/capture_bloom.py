"""One exact Bloom Sable pass, then same-IR old-parallel diagnostic only."""
from contextlib import ExitStack
from unittest.mock import patch
import argparse,copy,difflib,json,subprocess,sys
from xml.etree import ElementTree
from capture_common import begin,finish,failure,write,sha,canonical,load_module,save_diagram,save_preservation_input

p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--source-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--native',action='store_true');a=p.parse_args()
root,out,before,start=begin(a)
try:
    from model_unfolder.diagram import Diagram
    from model_unfolder.sable import sable
    from model_unfolder.preview import svg_views,_visual_hash,svg_to_png
    from model_unfolder.renderers.html import graph_engine,op_render
    helpers=load_module('s8_bloom_demonstration_helpers',root/'scripts/demonstrate_s8_unet.py')
    fixture_path=root/'tests/sable_test_corpus/bloom.json';fixture_pin=sha(fixture_path);fixture=json.loads(fixture_path.read_text());write(out/'fixture.json',fixture);write(out/'fixture-pin.json',{'path':str(fixture_path),'sha256':sha(fixture_path)})
    rendered=[];graphs=[];regions={};phase='actual_sable';original_html=Diagram.to_html;original_graph=graph_engine._render_graph_in_context;original_region=op_render.region_to_graph
    def capture_region(region,*args,**kwargs):
        graph=original_region(region,*args,**kwargs);regions[id(graph)]=(graph,copy.deepcopy(region));return graph
    def capture_graph(graph,*args,**kwargs):
        svg=original_graph(graph,*args,**kwargs);region=regions.get(id(graph),(None,None))[1]
        empty=[]
        for parallel in graph.parallels:
            for lane in parallel.norm_lanes():
                if lane.ids:continue
                source=lane.src or parallel.src
                for target in lane.dst or [parallel.dst]:
                    edge_match=bool(region is not None and any(edge.src==source and edge.dst==target for edge in region.edges))
                    empty.append({'source':source,'target':target,'canonical_region_edge':edge_match,'origin':'Region edge' if edge_match else 'explicit Graph empty-lane relation'})
        context=kwargs.get('context');graphs.append({'phase':phase,'view':args[2],'graph':canonical(graph),'region':canonical(region),'empty_lane_edges':empty,'svg_sha256':__import__('hashlib').sha256(svg.encode()).hexdigest(),'visual_hash':_visual_hash(svg),'block_path':[b.get('id') for b in context.block_stack] if context else []})
        return svg
    def capture_html(self,standalone=True):
        page=original_html(self,standalone=standalone)
        if standalone:rendered.append((self,page))
        return page
    with ExitStack() as stack:
        # Existing imported aliases and later imports both call this same observer.
        for module in list(sys.modules.values()):
            if getattr(module,'__name__','').startswith('model_unfolder.renderers.html') and getattr(module,'region_to_graph',None) is original_region:
                stack.enter_context(patch.object(module,'region_to_graph',capture_region))
        stack.enter_context(patch.object(graph_engine,'_render_graph_in_context',capture_graph));stack.enter_context(patch.object(Diagram,'to_html',capture_html))
        report=sable(fixture['config'],source=fixture.get('source','local'),render_images=False)
        assert rendered
        actual,page=rendered[-1];current_dir=out/'current';current_dir.mkdir();save_diagram(current_dir,actual,helpers);write(out/'sable-report.json',canonical(report));save_preservation_input(current_dir,actual,root,fixture_path);assert page==actual.to_html()
        assert all(par['src'] is not None for row in graphs if row['phase']=='actual_sable' for par in row['graph']['parallels']),'independent-input branch present; old-function isolation requires separate analysis'
        old_source=subprocess.check_output(['git','show','83140f1:model_unfolder/renderers/html/graph_engine.py'],cwd=root)
        old_path=out/'graph_engine_83140f1.py';old_path.write_bytes(old_source);old_module=load_module('model_unfolder.renderers.html._s8_old_parallel_diagnostic',old_path)
        diagnostic=Diagram(copy.deepcopy(actual.ir));diagnostic._mount_id=actual._mount_id;assert diagnostic.to_ir()==actual.to_ir()
        phase='old_parallel_only';old_dir=out/'old-parallel-only';old_dir.mkdir()
        with patch.object(graph_engine,'_draw_parallel',old_module._draw_parallel):old_page,_=save_diagram(old_dir,diagnostic,helpers)
    write(out/'graphs-and-regions.json',graphs)
    def distinct(html):
        rows=[];seen=set()
        for label,svg in svg_views(html):
            value=_visual_hash(svg)
            if value not in seen:rows.append((label,value,svg));seen.add(value)
        return rows
    current=distinct(page);old=distinct(old_page);assert [(l,h) for l,h,_ in current]==report.view_hashes
    viewdir=out/'views';viewdir.mkdir();deltas=[]
    for label,indexed in [('current',current),('old-parallel-only',old)]:
        for number,(name,value,svg) in enumerate(indexed):(viewdir/f'{label}-{number:02d}-{name}.svg').write_text(svg)
    for number in range(max(len(current),len(old))):
        c=current[number] if number<len(current) else None;o=old[number] if number<len(old) else None
        row={'index':number,'current_label':c[0] if c else None,'old_label':o[0] if o else None,'current_hash':c[1] if c else None,'old_hash':o[1] if o else None,'changed':c is None or o is None or c[:2]!=o[:2]};deltas.append(row)
        if c and o and row['changed']:
            def pretty(svg):
                tree=ElementTree.fromstring(svg);ElementTree.indent(tree);return ElementTree.tostring(tree,encoding='unicode').splitlines(True)
            (viewdir/f'{number:02d}.diff').write_text(''.join(difflib.unified_diff(pretty(o[2]),pretty(c[2]),fromfile='old-parallel-only',tofile='current')))
            if a.native:
                for prefix,value in [('old',o),('current',c)]:svg_to_png(value[2],str(viewdir/f'{number:02d}-{prefix}.png'),scale=1.0)
    write(out/'view-deltas.json',deltas)
    recovered=sorted(h for _,h,_ in old)==fixture['hash_signature'] and [{'label':l,'hash':h} for l,h,_ in sorted(old)]==fixture['view_signature']
    assert sha(fixture_path)==fixture_pin
    finish(root,out,before,start,{'status':'CAPTURED_REVIEW_REQUIRED','locked_views':fixture['view_signature'],'current_views':report.view_signature(),'old_parallel_recovers_all_locked_views':recovered,'changed_view_count':sum(row['changed'] for row in deltas),'canonical_graphs_and_regions':'graphs-and-regions.json','only_diagnostic_source_override':'83140f1 graph_engine._draw_parallel in memory; all other final source and typed IR unchanged','no_second_parse':True,'no_baseline_write':True})
except BaseException as error:
    failure(out,error);raise
