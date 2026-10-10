# Adapted from Kaminos models/trellis2/process-memory.py @ 5c9a2fc2939e0a28ac3560a1e9108e81ba245fa7.
# Source provenance retained; SF3D schema/name adaptation, not a new physical-capacity claim.
"""One uncapped owned-process sample using actual Darwin libproc RUSAGE_INFO_V4."""
import argparse,ctypes,errno,json,os,subprocess,sys,time

# Exact current SDK sys/resource.h ABI: UUID16 plus35 uint64 fields, in declaration order.
FIELDS='user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy serviced_energy interval_max_phys_footprint runnable_time'.split()
class RUsageInfoV4(ctypes.Structure):
    _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in FIELDS]

def sample_owned_processes(root_pid,run_id):
    report={'schema':'sf3d.process-memory-sample.v0','runId':run_id,'rootPid':root_pid,'atUnixMs':time.time()*1000,
        'status':'unavailable','effectiveRoute':'darwin-libproc-proc_pid_rusage/RUSAGE_INFO_V4','processes':[],
        'sampledAggregatePhysicalFootprintBytes':None,'meaning':'concurrent owned-process charged footprint; not whole-machine usage or summed lifetime peaks'}
    if sys.platform!='darwin':report['error']='Darwin libproc route unavailable on '+sys.platform;return report
    try:
        library=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        library.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(RUsageInfoV4)]
        library.proc_pid_rusage.restype=ctypes.c_int
        rows={}
        for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,comm='],text=True).splitlines():
            parts=line.strip().split(None,2)
            if len(parts)==3:rows[int(parts[0])]={'pid':int(parts[0]),'parentPid':int(parts[1]),'executable':parts[2]}
        if root_pid not in rows:report['error']='owned root process is absent';return report
        owned={root_pid}
        while True:
            children={pid for pid,row in rows.items() if row['parentPid'] in owned}
            expanded=owned|children
            if expanded==owned:break
            owned=expanded
        for pid in sorted(owned):
            info=RUsageInfoV4();error=library.proc_pid_rusage(pid,4,ctypes.byref(info))
            if error:
                missing={**rows[pid],'errno':ctypes.get_errno(),
                    'expectedProbeExit':rows[pid]['parentPid']==os.getpid() and rows[pid]['executable']=='/bin/ps'}
                # The tree snapshot and libproc reads are not atomic. A short
                # source-attestation git child can exit between them. Only an
                # ESRCH plus a fresh OS absence/zombie check proves retirement;
                # a live or inaccessible process remains missing coverage.
                if pid!=root_pid and missing['errno']==errno.ESRCH:
                    try:
                        check=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
                        state=check.stdout.strip()
                        evidence={'route':'ps-pid-status','returnCode':check.returncode,'status':state,'stderr':check.stderr}
                        missing['exitEvidence']=evidence
                        if not check.stderr and ((check.returncode==1 and not state) or
                                (check.returncode==0 and state.startswith('Z'))):
                            missing['exitedBeforeMeasurement']=True
                    except Exception as error:
                        missing['exitCheckError']=str(error)
                report.setdefault('unavailableProcesses',[]).append(missing);continue
            report['processes'].append({**rows[pid],'physicalFootprintBytes':info.phys_footprint,
                'residentBytes':info.resident_size,'kernelLifetimePeakPhysicalFootprintBytes':info.lifetime_max_phys_footprint,
                'processStartAbstime':info.proc_start_abstime,'observerProcess':pid==os.getpid()})
        if not any(row['pid']==root_pid for row in report['processes']):report['error']='root identity not observed by libproc';return report
        report['status']='observed';report['sampledAggregatePhysicalFootprintBytes']=sum(x['physicalFootprintBytes'] for x in report['processes'])
        report['observerIncluded']=True;report['abiSizeBytes']=ctypes.sizeof(RUsageInfoV4)
    except Exception as error:report['error']=str(error)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root-pid',type=int,required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    report=sample_owned_processes(a.root_pid,a.run_id);print(json.dumps(report));sys.exit(0 if report['status']=='observed' else 1)
