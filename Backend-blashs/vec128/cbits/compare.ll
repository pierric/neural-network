define cc10 void @vfcomp_oge(i64* %baseReg,
                             i64* %sp,
                             i64* %hp,
                             <4 x float> %r1, <4 x float> %r2)
{
  %1 = getelementptr inbounds i64, i64* %sp, i64 0
  %2 = load i64, i64* %1, align 8
  %r = fcmp oge <4 x float> %r1, %r2
  %s = zext <4 x i1> %r to <4 x i32>
  %cont = inttoptr i64 %2 to void (i64*,
                                   i64*,
                                   i64*,
                                   <4 x i32>)*
  tail call cc10 void %cont(i64* %baseReg,
                            i64* %sp,
                            i64* %hp,
                            <4 x i32> %s) noreturn
  ret void
}

define cc10 void @vselect(i64* %baseReg,
                          i64* %sp,
                          i64* %hp,
                          <4 x i32> %r1, <4 x float> %r2, <4 x float> %r3)
{
  %1 = getelementptr inbounds i64, i64* %sp, i64 0
  %2 = load i64, i64* %1, align 8
  %c = trunc <4 x i32> %r1 to <4 x i1>
  %rr = select <4 x i1> %c, <4 x float> %r2, <4 x float> %r3
  %cont = inttoptr i64 %2 to void (i64*,
                                   i64*,
                                   i64*,
                                   <4 x float>)*
  tail call cc10 void %cont(i64* %baseReg,
                            i64* %sp,
                            i64* %hp,
                            <4 x float> %rr) noreturn
  ret void
}
