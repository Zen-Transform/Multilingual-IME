from tqdm import tqdm
from multiprocessing import Pool
from typing import Callable


def multiprocessing(proccess_batch: Callable, batch_args: list[tuple], show_progress=False, preserve_order=True):
    """
    Multiprocessing wrapper for a function that takes a single argument.
    """
    with Pool() as pool:
        if show_progress:
            with tqdm(total=len(batch_args)) as pbar:
                def update_progress(*a):
                    pbar.update()
                
                async_result = []
                if preserve_order:
                    for index, args in enumerate(batch_args):
                        async_result.append((index, pool.apply_async(proccess_batch, args, callback=update_progress)))
                    
                    try:
                        results = [r.get() for i, r in sorted(async_result, key=lambda x: x[0])]
                    except Exception as e:
                        print(f"Error during multiprocessing: {e}")
                        raise
                else:
                    for args in batch_args:
                        async_result.append(pool.apply_async(proccess_batch, args, callback=update_progress))
                    
                    try:
                        results = [r.get() for r in async_result]
                    except Exception as e:
                        print(f"Error during multiprocessing: {e}")
                        raise         
        else:
            async_result = []
            if preserve_order:
                async_result = pool.starmap_async(proccess_batch, batch_args)

                try:
                    results = async_result.get()
                except Exception as e:
                    print(f"Error during multiprocessing: {e}")
                    raise  
            else:

                for args in batch_args:
                    async_result.append(pool.apply_async(proccess_batch, args))
                    
                try:
                    results = [r.get() for r in async_result]
                except Exception as e:
                    print(f"Error during multiprocessing: {e}")
                    raise   

    return results

def sum(a: int, b: int) -> int:
    return a + b    

if __name__ == "__main__":
    job_list = [('a', 'b'), ('c', 'd'), ('e', 'f'), ('g', 'h')]
    result = multiprocessing(sum, job_list, show_progress=False, preserve_order=True)
    print(result)
    print("=====================================")
    result = multiprocessing(sum, job_list, show_progress=False, preserve_order=False)
    print(result)
    print("=====================================")
    result = multiprocessing(sum, job_list, show_progress=True, preserve_order=True)
    print(result)
    print("=====================================")
    result = multiprocessing(sum, job_list, show_progress=True, preserve_order=True)
    print(result)
    print("=====================================")

